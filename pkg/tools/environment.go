package tools

import (
	"bytes"
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/tools/environment"
)

/* EnvironmentTool provides a base for all environment operations */
type EnvironmentTool struct {
	Tools []Tool
}

/* NewEnvironmentTool creates a new environment tool with all operations */
func NewEnvironmentTool() *EnvironmentTool {
	command := NewEnvironmentCommandTool()
	input := NewEnvironmentInputTool()

	return &EnvironmentTool{
		Tools: []Tool{
			{
				Tool: command.Tool,
				Use:  command.Use,
			},
			{
				Tool: input.Tool,
				Use:  input.Use,
			},
		},
	}
}

/* EnvironmentCommandTool implements a tool for executing commands */
type EnvironmentCommandTool struct {
	mcp.Tool
}

/* NewEnvironmentCommandTool creates a new tool for executing commands */
func NewEnvironmentCommandTool() *EnvironmentCommandTool {
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
	}
}

/* Use executes the command operation */
func (tool *EnvironmentCommandTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	builder := environment.NewBuilder()

	if builder == nil {
		return mcp.NewToolResultText("error creating builder"), nil
	}

	runner := environment.NewRunner(builder.Runtime)

	if runner == nil {
		return mcp.NewToolResultText("error creating runner"), nil
	}

	command := req.Params.Arguments["command"].(string)

	if command == "" {
		return mcp.NewToolResultText("missing required field: command"), nil
	}

	var stdoutBuf, stderrBuf bytes.Buffer

	if err := runner.ExecuteCommand(ctx, command, &stdoutBuf, &stderrBuf); err != nil {
		return mcp.NewToolResultText("error executing command"), err
	}

	output := stdoutBuf.String()

	if output == "" && stderrBuf.Len() > 0 {
		output = stderrBuf.String()
	}

	return mcp.NewToolResultText(output), nil
}

/* EnvironmentInputTool implements a tool for providing input */
type EnvironmentInputTool struct {
	mcp.Tool
}

/* NewEnvironmentInputTool creates a new tool for providing input */
func NewEnvironmentInputTool() *EnvironmentInputTool {
	return &EnvironmentInputTool{
		Tool: mcp.NewTool(
			"input",
			mcp.WithDescription(
				"A tool which gives you a full Linux terminal-based environment to interact with.",
			),
			mcp.WithString(
				"input",
				mcp.Description(
					"Valid input to pass to the environment, used for interactive sessions.",
				),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the input operation */
func (tool *EnvironmentInputTool) Use(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	builder := environment.NewBuilder()

	if builder == nil {
		return mcp.NewToolResultText("error creating builder"), nil
	}

	runner := environment.NewRunner(builder.Runtime)

	if runner == nil {
		return mcp.NewToolResultText("error creating runner"), nil
	}

	input := req.Params.Arguments["input"].(string)

	if input == "" {
		return mcp.NewToolResultText("missing required field: input"), nil
	}

	if input[len(input)-1] != '\n' {
		input += "\n"
	}

	output, err := runner.SendInput(input)

	if err != nil {
		return mcp.NewToolResultText("error sending input"), err
	}

	return mcp.NewToolResultText(output), nil
}
