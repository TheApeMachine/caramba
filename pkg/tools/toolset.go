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
	Tools []CompositeTool
}

type ToolsetOption func(*Toolset)

func NewToolset(opts ...ToolsetOption) *Toolset {
	toolset := &Toolset{
		Tools: []CompositeTool{},
	}

	for _, opt := range opts {
		opt(toolset)
	}

	return toolset
}

func WithTools(tool CompositeTool) ToolsetOption {
	return func(toolset *Toolset) {
		toolset.Tools = append(toolset.Tools, tool)
	}
}

// ToMCP returns the MCP tool definitions for all tools in the toolset
func (t *Toolset) ToMCP() []mcp.Tool {
	var tools []mcp.Tool
	for _, tool := range t.Tools {
		tools = append(tools, tool.ToMCP()...)
	}
	return tools
}
