package task

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
)

type Send struct {
	mcp.Tool
}

/*
NewSend creates a new tool for sending tasks to an agent.
*/
func NewSend() *Send {
	return &Send{
		Tool: mcp.NewTool(
			"send_task",
			mcp.WithDescription("A tool which can send tasks to an agent."),
			mcp.WithString(
				"task",
				mcp.Description("The task to send to the agent."),
				mcp.Required(),
			),
			mcp.WithString(
				"agent",
				mcp.Description("The agent to send the task to."),
				mcp.Required(),
			),
		),
	}
}

func (tool *Send) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("editor write not implemented"), nil
}
