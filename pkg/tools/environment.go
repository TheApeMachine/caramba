package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/tools/environment"
)

/* EnvironmentTool provides a base for all environment operations */
type EnvironmentTool struct {
	operations map[string]ToolType
}

/* NewEnvironmentTool creates a new environment tool with all operations */
func NewEnvironmentTool() *EnvironmentTool {
	command := NewEnvironmentCommandTool()
	input := NewEnvironmentInputTool()

	return &EnvironmentTool{
		operations: map[string]ToolType{
			"command": {command.Tool, command.Use},
			"input":   {input.Tool, input.Use},
		},
	}
}

/* ToMCP returns all environment tool definitions */
func (tool *EnvironmentTool) ToMCP() []ToolType {
	tools := make([]ToolType, 0)

	for _, tool := range tool.operations {
		tools = append(tools, tool)
	}

	return tools
}

/* EnvironmentCommandTool implements a tool for executing commands */
type EnvironmentCommandTool struct {
	mcp.Tool
	builder *environment.Builder
	runner  *environment.Runner
}

/* NewEnvironmentCommandTool creates a new tool for executing commands */
func NewEnvironmentCommandTool() *EnvironmentCommandTool {
	builder := environment.NewBuilder()
	if builder == nil {
		return nil
	}

	runner := environment.NewRunner(builder.Runtime)
	if runner == nil {
		return nil
	}

	return &EnvironmentCommandTool{
		Tool: mcp.NewTool(
			"command",
			mcp.WithDescription("A tool which gives you a full Linux terminal-based environment to interact with."),
			mcp.WithString(
				"command",
				mcp.Description("A valid bash command to execute in the environment."),
				mcp.Required(),
			),
		),
		builder: builder,
		runner:  runner,
	}
}

/* Use executes the command operation */
func (tool *EnvironmentCommandTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	// TODO: Implement actual command execution using builder/runner
	return mcp.NewToolResultText("Hello, world!"), nil
}

func (tool *EnvironmentCommandTool) ID() string {
	return "environment_command"
}

/* EnvironmentInputTool implements a tool for providing input */
type EnvironmentInputTool struct {
	mcp.Tool
	builder *environment.Builder
	runner  *environment.Runner
}

/* NewEnvironmentInputTool creates a new tool for providing input */
func NewEnvironmentInputTool() *EnvironmentInputTool {
	builder := environment.NewBuilder()
	if builder == nil {
		return nil
	}

	runner := environment.NewRunner(builder.Runtime)
	if runner == nil {
		return nil
	}

	return &EnvironmentInputTool{
		Tool: mcp.NewTool(
			"input",
			mcp.WithDescription("A tool which gives you a full Linux terminal-based environment to interact with."),
			mcp.WithString(
				"input",
				mcp.Description("Valid input to pass to the environment, used for interactive sessions."),
				mcp.Required(),
			),
		),
		builder: builder,
		runner:  runner,
	}
}

/* Use executes the input operation */
func (tool *EnvironmentInputTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	// TODO: Implement actual input handling using builder/runner
	return mcp.NewToolResultText("Hello, world!"), nil
}

func (tool *EnvironmentInputTool) ID() string {
	return "environment_input"
}
