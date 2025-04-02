package tools

import (
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
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

func (builder *ToolBuilder) Schema() *Tool {
	return builder.Tool
}

func WithMCP(mcp mcp.Tool) ToolOption {
	return func(builder *ToolBuilder) {
		builder.mcp = &mcp
	}
}
