package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/system"
)

func init() {
	provider.RegisterTool("system")
}

type SystemTool struct {
	*ToolBuilder
	pctx   context.Context
	ctx    context.Context
	cancel context.CancelFunc
	hub    *system.Hub
}

type SystemToolOption func(*SystemTool)

func NewSystemTool(opts ...SystemToolOption) *SystemTool {
	ctx, cancel := context.WithCancel(context.Background())

	tool := &SystemTool{
		ToolBuilder: NewToolBuilder(),
		ctx:         ctx,
		cancel:      cancel,
	}

	for _, opt := range opts {
		opt(tool)
	}

	return tool
}

func WithCancel(ctx context.Context) SystemToolOption {
	return func(tool *SystemTool) {
		tool.pctx = ctx
	}
}

func (tool *SystemTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("system.SystemTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		for {
			select {
			case <-tool.pctx.Done():
				errnie.Debug("system.SystemTool.Generate: parent context done")
				tool.cancel()
				return
			case <-tool.ctx.Done():
				errnie.Debug("system.SystemTool.Generate: context done")
				return
			case artifact := <-buffer:
				for _, f := range fn {
					out <- f(artifact)
				}
			}
		}
	}()

	return out
}

type SystemInspectTool struct {
	*SystemTool
	Schema *provider.Tool
}

func NewSystemInspectTool() *SystemInspectTool {
	inspectTool := provider.NewTool(
		provider.WithFunction("system_inspect", "A tool for inspecting the system."),
		provider.WithProperty("scope", "string", "The scope of the inspection.", []any{"agents", "topics"}),
		provider.WithRequired("scope"),
	)

	return &SystemInspectTool{
		SystemTool: NewSystemTool(),
		Schema:     inspectTool,
	}
}

func (tool *SystemInspectTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SystemTool.Generate(buffer, tool.fn)
}

func (tool *SystemInspectTool) fn(artifact *datura.Artifact) *datura.Artifact {
	return artifact
}

// ToMCP returns the MCP tool definitions for the SystemInspectTool
func (tool *SystemInspectTool) ToMCP() []mcp.Tool {
	return []mcp.Tool{tool.Schema.ToMCP()}
}

type SystemOptimizeTool struct {
	*SystemTool
	Schema *provider.Tool
}

func NewSystemOptimizeTool() *SystemOptimizeTool {
	optimizeTool := provider.NewTool(
		provider.WithFunction("system_optimize", "A tool for optimizing your performance and behavior."),
		provider.WithProperty("operation", "string", "The operation to perform.", []any{"inspect", "adjust"}),
		provider.WithProperty("temperature", "number", "The temperature parameter to adjust.", nil),
		provider.WithProperty("topP", "number", "The top_p parameter to adjust.", nil),
		provider.WithRequired("operation"),
	)

	return &SystemOptimizeTool{
		SystemTool: NewSystemTool(),
		Schema:     optimizeTool,
	}
}

func (tool *SystemOptimizeTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SystemTool.Generate(buffer, tool.fn)
}

func (tool *SystemOptimizeTool) fn(artifact *datura.Artifact) *datura.Artifact {
	return artifact
}

// ToMCP returns the MCP tool definitions for the SystemOptimizeTool
func (tool *SystemOptimizeTool) ToMCP() []mcp.Tool {
	return []mcp.Tool{tool.Schema.ToMCP()}
}

type SystemMessageTool struct {
	*SystemTool
	Schema *provider.Tool
}

func NewSystemMessageTool() *SystemMessageTool {
	messageTool := provider.NewTool(
		provider.WithFunction("system_message", "A tool for sending messages to other agents and topics."),
		provider.WithProperty("to", "string", "The name of the agent or topic to send a message to", nil),
		provider.WithProperty("message", "string", "The message to send to the agent or topic", nil),
		provider.WithRequired("to", "message"),
	)

	return &SystemMessageTool{
		SystemTool: NewSystemTool(),
		Schema:     messageTool,
	}
}

func (tool *SystemMessageTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.SystemTool.Generate(buffer, tool.fn)
}

func (tool *SystemMessageTool) fn(artifact *datura.Artifact) *datura.Artifact {
	return artifact
}

// ToMCP returns the MCP tool definitions for the SystemMessageTool
func (tool *SystemMessageTool) ToMCP() []mcp.Tool {
	return []mcp.Tool{tool.Schema.ToMCP()}
}
