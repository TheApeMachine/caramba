package tools

import (
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/stream"
)

type ToolInterface interface {
	stream.Generator
	ToMCP() []mcp.Tool
}

type CompositeTool interface {
	ToolInterface
}

// var (
// 	SystemCompositeTool CompositeTool = NewSystemTool()
// )

type Toolset struct {
	tools []CompositeTool
}

type ToolsetOption func(*Toolset)

func NewToolset(opts ...ToolsetOption) *Toolset {
	toolset := &Toolset{
		tools: []CompositeTool{},
	}

	for _, opt := range opts {
		opt(toolset)
	}

	return toolset
}

func WithTools(tool CompositeTool) ToolsetOption {
	return func(toolset *Toolset) {
		toolset.tools = append(toolset.tools, tool)
	}
}

// ToMCP returns the MCP tool definitions for all tools in the toolset
func (t *Toolset) ToMCP() []mcp.Tool {
	var tools []mcp.Tool
	for _, tool := range t.tools {
		tools = append(tools, tool.ToMCP()...)
	}
	return tools
}
