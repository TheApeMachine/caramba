package core

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

// IterationManager handles agent iteration logic.
type IterationManager struct {
	logger            *output.Logger
	hub               *hub.Queue
	agent             Agent
	toolManager       *ToolManager
	responseProcessor *ResponseProcessor
	workflowManager   *WorkflowManager
}

// NewIterationManager creates a new IterationManager.
func NewIterationManager(
	agent Agent,
) *IterationManager {
	return &IterationManager{
		logger:            output.NewLogger(),
		hub:               hub.NewQueue(),
		agent:             agent,
		toolManager:       NewToolManager(),
		responseProcessor: NewResponseProcessor(),
		workflowManager:   NewWorkflowManager(),
	}
}

func (im *IterationManager) SetWorkflow(workflow *Workflow) {
	im.workflowManager.SetWorkflow(workflow)
}

// Run handles the iteration loop, either in streaming or non-streaming mode.
func (im *IterationManager) Run(
	ctx context.Context,
	initialMsg LLMMessage,
	params *LLMParams,
	streaming bool,
) error {
	im.logger.Log(fmt.Sprintf("Running iteration manager for agent %s", im.agent.Name()))

	iteration := 0
	iterMsg := LLMMessage{
		Role:    initialMsg.Role,
		Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, initialMsg.Content),
	}

	for iteration < im.agent.IterationLimit() {
		// Add the current message to the conversation.
		params.Messages = append(params.Messages, iterMsg)

		// Create a status message about the iteration
		iterationStr := fmt.Sprintf("Iteration %d/%d", iteration+1, im.agent.IterationLimit())

		im.hub.Add(hub.NewEvent(
			im.agent.Name(),
			"metrics",
			"assistant",
			hub.EventTypeStatus,
			iterationStr,
			map[string]string{},
		))

		var iterationResponse string
		var err error

		if streaming {
			iterationResponse, err = im.handleStreamingIteration(
				ctx, *params, im.agent.LLM(),
			)
		} else {
			iterationResponse, err = im.handleNonStreamingIteration(
				ctx, *params, im.agent.LLM(),
			)
		}

		if err != nil {
			return err
		}

		// Next iteration
		iteration++
		iterMsg = LLMMessage{
			Role:    "assistant",
			Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, iterationResponse),
		}
	}

	return nil
}

// handleNonStreamingIteration processes a single non-streaming iteration.
func (im *IterationManager) handleNonStreamingIteration(
	ctx context.Context,
	params LLMParams,
	llm LLMProvider,
) (string, error) {
	// Send thinking status
	im.hub.Add(hub.NewEvent(
		im.agent.Name(),
		"ui",
		"assistant",
		hub.EventTypeStatus,
		"thinking",
		map[string]string{},
	))

	res := llm.GenerateResponse(ctx, params)

	if res.Error != nil {
		im.hub.Add(hub.NewEvent(
			im.agent.Name(),
			"ui",
			"assistant",
			hub.EventTypeError,
			res.Error.Error(),
			map[string]string{},
		))
		im.logError("Failed to send error status", res.Error)
		return "", res.Error
	}

	var responseWithToolResults strings.Builder
	responseWithToolResults.WriteString(res.Content)

	// Send the complete response to the UI
	im.hub.Add(hub.NewEvent(
		im.agent.Name(),
		"ui",
		"assistant",
		hub.EventTypeMessage,
		res.Content,
		map[string]string{},
	))

	// Handle tool calls if any
	if len(res.ToolCalls) > 0 {
		im.hub.Add(hub.NewEvent(
			im.agent.Name(),
			"metrics",
			"assistant",
			hub.EventTypeToolCall,
			fmt.Sprintf("%d tool calls", len(res.ToolCalls)),
			map[string]string{},
		))

		for _, toolCall := range res.ToolCalls {
			im.hub.Add(hub.NewEvent(
				im.agent.Name(),
				"actions",
				"assistant",
				hub.EventTypeToolCall,
				toolCall.Name,
				map[string]string{},
			))
		}

		// Execute tools and add results to the response
		toolResults := im.toolManager.ExecuteToolCalls(ctx, res.ToolCalls)
		responseWithToolResults.WriteString(toolResults)
	} else {
		im.hub.Add(hub.NewEvent(
			im.agent.Name(),
			"ui",
			"assistant",
			hub.EventTypeStatus,
			"complete",
			map[string]string{},
		))
	}

	return responseWithToolResults.String(), nil
}

// handleStreamingIteration processes a single streaming iteration.
func (im *IterationManager) handleStreamingIteration(
	ctx context.Context,
	params LLMParams,
	llm LLMProvider,
) (string, error) {
	// Instead of direct output, send status via hub
	im.hub.Add(hub.NewEvent(
		im.agent.Name(),
		"ui",
		"assistant",
		hub.EventTypeStatus,
		"streaming",
		map[string]string{},
	))

	// Accumulate the chunks
	var streamedResponse strings.Builder
	toolCallsFound := false
	var toolCalls []ToolCall

	// Process the stream
	streamChan := llm.StreamResponse(ctx, params)
	for chunk := range streamChan {
		if chunk.Error != nil {
			im.hub.Add(hub.NewEvent(
				im.agent.Name(),
				"ui",
				"assistant",
				hub.EventTypeError,
				chunk.Error.Error(),
				map[string]string{
					"model": params.Model,
				},
			))
			im.logError("Failed to send stream error", chunk.Error)
			return "", chunk.Error
		}

		// Handle tool calls in the chunk
		if len(chunk.ToolCalls) > 0 {
			toolCallsFound = true
			toolCalls = chunk.ToolCalls
			im.toolManager.LogToolCalls(toolCalls, ctx)
		} else if chunk.Content != "" {
			// Process and display content
			content := im.responseProcessor.ProcessChunkContent(chunk.Content)
			if content != "" {
				// Send content chunk directly to UI via hub instead of console
				im.hub.Add(hub.NewEvent(
					im.agent.Name(),
					"ui",
					"assistant",
					hub.EventTypeChunk,
					content,
					map[string]string{},
				))
				streamedResponse.WriteString(content)
			}
		}
	}

	// Signal that streaming is complete
	im.hub.Add(hub.NewEvent(
		im.agent.Name(),
		"ui",
		"assistant",
		hub.EventTypeStatus,
		"complete",
		map[string]string{},
	))

	// Execute tool calls if found
	if toolCallsFound {
		toolResults := im.toolManager.ExecuteToolCalls(ctx, toolCalls)
		streamedResponse.WriteString(toolResults)
	}

	return streamedResponse.String(), nil
}

// logError logs an error to the log file
func (im *IterationManager) logError(context string, err error) {
	im.logger.Log(fmt.Sprintf("[ERROR] %s: %v", context, err))
}
