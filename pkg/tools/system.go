package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type SystemTool struct {
	*ToolBuilder
	pctx   context.Context
	ctx    context.Context
	cancel context.CancelFunc
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
}

func NewSystemInspectTool() *SystemInspectTool {
	inspectTool := mcp.NewTool(
		"system_inspect",
		mcp.WithDescription("A tool for inspecting the system."),
		mcp.WithString(
			"scope",
			mcp.Description("The scope of the inspection."),
			mcp.Enum("agents", "topics"),
			mcp.Required(),
		),
	)

	sit := &SystemInspectTool{
		SystemTool: NewSystemTool(),
	}

	sit.ToolBuilder.mcp = &inspectTool
	return sit
}

func (tool *SystemInspectTool) ID() string {
	return "system_inspect"
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

func (tool *SystemInspectTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

type SystemOptimizeTool struct {
	*SystemTool
}

func NewSystemOptimizeTool() *SystemOptimizeTool {
	optimizeTool := mcp.NewTool(
		"system_optimize",
		mcp.WithDescription("A tool for optimizing your performance and behavior."),
		mcp.WithString(
			"operation",
			mcp.Description("The operation to perform."),
			mcp.Enum("inspect", "adjust"),
			mcp.Required(),
		),
		mcp.WithNumber(
			"temperature",
			mcp.Description("The temperature parameter to adjust."),
		),
		mcp.WithNumber(
			"topP",
			mcp.Description("The top_p parameter to adjust."),
		),
	)

	sot := &SystemOptimizeTool{
		SystemTool: NewSystemTool(),
	}

	sot.ToolBuilder.mcp = &optimizeTool
	return sot
}

func (tool *SystemOptimizeTool) ID() string {
	return "system_optimize"
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
func (tool *SystemOptimizeTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

type SystemMessageTool struct {
	*SystemTool
}

func NewSystemMessageTool() *SystemMessageTool {
	messageTool := mcp.NewTool(
		"system_message",
		mcp.WithDescription("A tool for sending messages to other agents and topics."),
		mcp.WithString(
			"to",
			mcp.Description("The name of the agent or topic to send a message to"),
			mcp.Required(),
		),
		mcp.WithString(
			"message",
			mcp.Description("The message to send to the agent or topic"),
			mcp.Required(),
		),
	)

	smt := &SystemMessageTool{
		SystemTool: NewSystemTool(),
	}

	smt.ToolBuilder.mcp = &messageTool
	return smt
}

func (tool *SystemMessageTool) ID() string {
	return "system_message"
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
func (tool *SystemMessageTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}
