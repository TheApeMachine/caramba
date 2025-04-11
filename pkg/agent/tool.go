package agent

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
)

type ToolWrapper struct {
	operations map[string]mcp.Tool
}

func NewToolWrapper() *ToolWrapper {
	return &ToolWrapper{
		operations: map[string]mcp.Tool{
			"instruction": NewInstructionTool().Tool,
		},
	}
}

type InstructionTool struct {
	mcp.Tool
}

func NewInstructionTool() *InstructionTool {
	return &InstructionTool{}
}

func (tool *InstructionTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Operation not implemented"), nil
}
