package prompts

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
)

// MCPHandler handles MCP requests for prompts
type MCPHandler struct {
	manager PromptManager
}

// NewMCPHandler creates a new MCPHandler
func NewMCPHandler(manager PromptManager) *MCPHandler {
	return &MCPHandler{
		manager: manager,
	}
}

// HandleListPrompts handles the prompts/list request
func (h *MCPHandler) HandleListPrompts(ctx context.Context, req *mcp.ListPromptsRequest) (*mcp.ListPromptsResult, error) {
	prompts, err := h.manager.List(ctx)
	if err != nil {
		return nil, err
	}

	// Convert prompts to MCP format
	mcpPrompts := make([]mcp.Prompt, len(prompts))
	for i, p := range prompts {
		mcpPrompts[i] = mcp.NewPrompt(p.Name,
			mcp.WithPromptDescription(p.Description),
		)
	}

	return mcp.NewListPromptsResult(mcpPrompts, ""), nil
}

// HandleGetPrompt handles the prompts/get request
func (h *MCPHandler) HandleGetPrompt(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	// Get all prompts and find the one with matching name
	prompts, err := h.manager.List(ctx)
	if err != nil {
		return nil, err
	}

	var prompt *Prompt
	for _, p := range prompts {
		if p.Name == req.Params.Name {
			prompt = &p
			break
		}
	}

	if prompt == nil {
		return nil, ErrorPromptNotFound{ID: req.Params.Name}
	}

	// Convert prompt content to MCP format
	messages := []mcp.PromptMessage{
		mcp.NewPromptMessage(mcp.RoleUser, mcp.NewTextContent(prompt.Content)),
	}

	// If it's a multi-step prompt, get the steps
	if prompt.Type == MultiStepPrompt {
		steps, err := h.manager.GetSteps(ctx, prompt.ID)
		if err != nil {
			return nil, err
		}

		// Add each step as a message
		for _, step := range steps {
			messages = append(messages, mcp.NewPromptMessage(
				mcp.RoleUser,
				mcp.NewTextContent(step.Content),
			))
		}
	}

	return mcp.NewGetPromptResult(prompt.Description, messages), nil
}

// HandleGetPromptSteps handles the prompts/getSteps request
func (h *MCPHandler) HandleGetPromptSteps(ctx context.Context, req *mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	// Get all prompts and find the one with matching name
	prompts, err := h.manager.List(ctx)
	if err != nil {
		return nil, err
	}

	var prompt *Prompt
	for _, p := range prompts {
		if p.Name == req.Params.Name {
			prompt = &p
			break
		}
	}

	if prompt == nil {
		return nil, ErrorPromptNotFound{ID: req.Params.Name}
	}

	if prompt.Type != MultiStepPrompt {
		return nil, ErrorInvalidPromptType{ID: prompt.ID, Type: prompt.Type}
	}

	steps, err := h.manager.GetSteps(ctx, prompt.ID)
	if err != nil {
		return nil, err
	}

	// Convert steps to messages
	messages := make([]mcp.PromptMessage, len(steps))
	for i, step := range steps {
		messages[i] = mcp.NewPromptMessage(
			mcp.RoleUser,
			mcp.NewTextContent(step.Content),
		)
	}

	return mcp.NewGetPromptResult(prompt.Description, messages), nil
}
