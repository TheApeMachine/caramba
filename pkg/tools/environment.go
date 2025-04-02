package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools/environment"
)

/*
EnvironmentTool provides a sandboxed Linux environment for executing commands.

It uses Docker containers to create an isolated environment for running commands
and interacting with a full Linux terminal. The environment is managed through
a builder for setup and a runner for command execution.

Future enhancements will include GVisor for additional sandboxing capabilities.
*/
type EnvironmentTool struct {
	*ToolBuilder
	pctx    context.Context
	ctx     context.Context
	cancel  context.CancelFunc
	builder *environment.Builder
	runner  *environment.Runner
}

type EnvironmentToolOption func(*EnvironmentTool)

/*
NewEnvironmentTool creates a new Environment tool instance.

It initializes a Docker container environment with necessary runtime components
and sets up a buffered stream for processing commands. Returns nil if either
the builder or runner initialization fails.
*/
func NewEnvironmentTool(opts ...EnvironmentToolOption) *EnvironmentTool {
	errnie.Debug("environment.NewEnvironmentTool")

	// Setup the docker container for the agent's environment tool.
	builder := environment.NewBuilder()
	if builder == nil {
		return nil
	}

	runner := environment.NewRunner(builder.Runtime)
	if runner == nil {
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	tool := &EnvironmentTool{
		ToolBuilder: NewToolBuilder(),
		ctx:         ctx,
		cancel:      cancel,
		builder:     builder,
		runner:      runner,
	}

	for _, opt := range opts {
		opt(tool)
	}

	return tool
}

func (tool *EnvironmentTool) ID() string {
	return "environment"
}

func WithEnvironmentCancel(ctx context.Context) EnvironmentToolOption {
	return func(tool *EnvironmentTool) {
		tool.pctx = ctx
	}
}

func (tool *EnvironmentTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("environment.EnvironmentTool.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		for {
			select {
			case <-tool.pctx.Done():
				errnie.Debug("environment.EnvironmentTool.Generate: parent context done")
				tool.cancel()
				return
			case <-tool.ctx.Done():
				errnie.Debug("environment.EnvironmentTool.Generate: context done")
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

// EnvironmentCommandTool implements a tool for executing commands in the environment
type EnvironmentCommandTool struct {
	*EnvironmentTool
}

// NewEnvironmentCommandTool creates a new tool for executing commands
func NewEnvironmentCommandTool() *EnvironmentCommandTool {
	// Create MCP tool definition based on schema from config.yml
	commandTool := mcp.NewTool(
		"command",
		mcp.WithDescription("A tool which gives you a full Linux terminal-based environment to interact with. You are directly connected to stdin, stdout, and stderr."),
		mcp.WithString(
			"command",
			mcp.Description("A valid bash command to execute in the environment."),
			mcp.Required(),
		),
	)

	ect := &EnvironmentCommandTool{
		EnvironmentTool: NewEnvironmentTool(),
	}

	ect.ToolBuilder.mcp = &commandTool
	return ect
}

func (tool *EnvironmentCommandTool) ID() string {
	return "environment_command"
}

// Generate processes the command execution operation
func (tool *EnvironmentCommandTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.runner.Generate(buffer)
}

// ToMCP returns the MCP tool definitions for the EnvironmentCommandTool
func (tool *EnvironmentCommandTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}

// EnvironmentInputTool implements a tool for providing input to the environment
type EnvironmentInputTool struct {
	*EnvironmentTool
}

// NewEnvironmentInputTool creates a new tool for providing input
func NewEnvironmentInputTool() *EnvironmentInputTool {
	// Create MCP tool definition based on schema from config.yml
	inputTool := mcp.NewTool(
		"input",
		mcp.WithDescription("A tool which gives you a full Linux terminal-based environment to interact with. You are directly connected to stdin, stdout, and stderr."),
		mcp.WithString(
			"input",
			mcp.Description("Valid input to pass to the environment, used for interactive sessions."),
			mcp.Required(),
		),
	)

	eit := &EnvironmentInputTool{
		EnvironmentTool: NewEnvironmentTool(),
	}

	eit.ToolBuilder.mcp = &inputTool
	return eit
}

func (tool *EnvironmentInputTool) ID() string {
	return "environment_input"
}

// Generate processes the input provision operation
func (tool *EnvironmentInputTool) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	return tool.runner.Generate(buffer)
}

// ToMCP returns the MCP tool definitions for the EnvironmentInputTool
func (tool *EnvironmentInputTool) ToMCP() mcp.Tool {
	return *tool.ToolBuilder.mcp
}
