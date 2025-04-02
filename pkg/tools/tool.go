package tools

import (
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/utils"
)

type ToolBuilder struct {
	mcp *mcp.Tool
	*Tool
}

type ToolOption func(*ToolBuilder)

func NewToolBuilder(opts ...ToolOption) *ToolBuilder {
	var (
		cpnp = utils.NewCapnp()
		tool Tool
		err  error
	)

	if tool, err = NewRootTool(cpnp.Seg); errnie.Error(err) != nil {
		return nil
	}

	builder := &ToolBuilder{
		Tool: &tool,
	}

	for _, opt := range opts {
		opt(builder)
	}

	return builder
}

func (builder *ToolBuilder) Schema() *provider.Tool {
	if builder.mcp != nil {
		// If we have an MCP tool, return a provider.Tool that wraps it
		return provider.NewTool(
			provider.WithFunction(builder.mcp.Name, builder.mcp.Description),
		)
	}
	return nil
}

func WithMCP(mcp mcp.Tool) ToolOption {
	return func(builder *ToolBuilder) {
		builder.mcp = &mcp
	}
}
