package interfaces

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
)

type Tool interface {
	Use(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error)
}
