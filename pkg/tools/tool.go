package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
)

type ToolType struct {
	Tool   mcp.Tool
	Use    func(ctx context.Context, artifact *datura.ArtifactBuilder) *datura.ArtifactBuilder
	UseMCP func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error)
}
