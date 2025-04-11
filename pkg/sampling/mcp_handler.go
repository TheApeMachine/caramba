package sampling

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
)

// MCPHandler handles MCP requests for sampling operations
type MCPHandler struct {
	manager SamplingManager
}

// NewMCPHandler creates a new MCPHandler
func NewMCPHandler(manager SamplingManager) *MCPHandler {
	return &MCPHandler{
		manager: manager,
	}
}

// HandleCreateMessage handles the sampling/createMessage request
func (h *MCPHandler) HandleCreateMessage(ctx context.Context, req *mcp.CreateMessageRequest) (*mcp.CreateMessageResult, error) {
	// Convert MCP preferences to our format
	prefs := ModelPreferences{
		Temperature: req.Params.Temperature,
		MaxTokens:   req.Params.MaxTokens,
		Stop:        req.Params.StopSequences,
	}

	// Convert MCP messages to our format
	messages := make([]Message, len(req.Params.Messages))
	for i, msg := range req.Params.Messages {
		content := ""
		if textContent, ok := msg.Content.(*mcp.TextContent); ok {
			content = textContent.Text
		}
		messages[i] = Message{
			Role:    string(msg.Role),
			Content: content,
		}
	}

	// Create sampling options
	opts := SamplingOptions{
		ModelPreferences: prefs,
		Context: &Context{
			Messages: messages,
		},
		Stream: false,
	}

	// Create the message
	result, err := h.manager.CreateMessage(ctx, req.Params.SystemPrompt, opts)
	if err != nil {
		return nil, err
	}

	// Convert our result to MCP format
	content := mcp.NewTextContent(result.Message.Content)
	return &mcp.CreateMessageResult{
		SamplingMessage: mcp.SamplingMessage{
			Role:    mcp.Role(result.Message.Role),
			Content: content,
		},
		Model: "default", // TODO: Get actual model name
	}, nil
}

// HandleStreamMessage handles streaming message creation
func (h *MCPHandler) HandleStreamMessage(ctx context.Context, req *mcp.CreateMessageRequest) (<-chan *mcp.CreateMessageResult, error) {
	// Convert MCP preferences to our format
	prefs := ModelPreferences{
		Temperature: req.Params.Temperature,
		MaxTokens:   req.Params.MaxTokens,
		Stop:        req.Params.StopSequences,
	}

	// Convert MCP messages to our format
	messages := make([]Message, len(req.Params.Messages))
	for i, msg := range req.Params.Messages {
		content := ""
		if textContent, ok := msg.Content.(*mcp.TextContent); ok {
			content = textContent.Text
		}
		messages[i] = Message{
			Role:    string(msg.Role),
			Content: content,
		}
	}

	// Create sampling options
	opts := SamplingOptions{
		ModelPreferences: prefs,
		Context: &Context{
			Messages: messages,
		},
		Stream: true,
	}

	// Create the streaming channel
	resultChan, err := h.manager.StreamMessage(ctx, req.Params.SystemPrompt, opts)
	if err != nil {
		return nil, err
	}

	// Create MCP result channel
	mcpResultChan := make(chan *mcp.CreateMessageResult)

	// Convert sampling results to MCP results
	go func() {
		defer close(mcpResultChan)

		for result := range resultChan {
			content := mcp.NewTextContent(result.Message.Content)
			mcpResultChan <- &mcp.CreateMessageResult{
				SamplingMessage: mcp.SamplingMessage{
					Role:    mcp.Role(result.Message.Role),
					Content: content,
				},
				Model: "default", // TODO: Get actual model name
			}
		}
	}()

	return mcpResultChan, nil
}
