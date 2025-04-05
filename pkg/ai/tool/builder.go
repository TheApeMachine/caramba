package tool

import (
	"capnproto.org/go/capnp/v3"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools"
)

type ToolBuilder struct {
	*Tool
	MCP []tools.ToolType
}

type ToolBuilderOption func(*ToolBuilder) error

// New creates a new tool with the provided options
func New(options ...ToolBuilderOption) *ToolBuilder {
	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		tool  Tool
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if tool, err = NewRootTool(seg); errnie.Error(err) != nil {
		return nil
	}

	builder := &ToolBuilder{
		Tool: &tool,
	}

	// Apply all options
	for _, option := range options {
		if err := option(builder); errnie.Error(err) != nil {
			return nil
		}
	}

	return builder
}

func (tool *ToolBuilder) ToMCP() *ToolBuilder {
	name, err := tool.Name()

	if errnie.Error(err) != nil {
		return nil
	}

	switch name {
	case "azure":
		tool.MCP = tools.NewAzureTool().ToMCP()
	case "browser":
		tool.MCP = tools.NewBrowserTool().ToMCP()
	case "editor":
		tool.MCP = tools.NewEditorTool().ToMCP()
	case "environment":
		tool.MCP = tools.NewEnvironmentTool().ToMCP()
	case "github":
		tool.MCP = tools.NewGithubTool().ToMCP()
	case "memory":
		tool.MCP = tools.NewMemoryTool().ToMCP()
	case "slack":
		tool.MCP = tools.NewSlackTool().ToMCP()
	case "system":
		tool.MCP = tools.NewSystemTool().ToMCP()
	case "trengo":
		tool.MCP = tools.NewTrengoTool().ToMCP()
	}

	return tool
}

// WithName sets the tool's name
func WithName(name string) ToolBuilderOption {
	return func(t *ToolBuilder) error {
		return errnie.Error(t.SetName(name))
	}
}
