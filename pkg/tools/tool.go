package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
)

type Tool struct {
	Tool mcp.Tool
	Use  func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error)
}
