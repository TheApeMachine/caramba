package tools

import (
	"bytes"
	"context"
	"errors"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
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
			"command": {command.Tool, command.Use, command.UseMCP},
			"input":   {input.Tool, input.Use, input.UseMCP},
		},
	}
}

func (tool *EnvironmentTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	toolName := datura.GetMetaValue[string](artifact, "tool")
	return tool.operations[toolName].Use(ctx, artifact)
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	builder := environment.NewBuilder()

	if builder == nil {
		return nil
	}

	runner := environment.NewRunner(builder.Runtime)

	if runner == nil {
		return nil
	}

	command := datura.GetMetaValue[string](artifact, "command")

	if command == "" {
		return datura.New(
			datura.WithRole(datura.ArtifactRoleTool),
			datura.WithScope(datura.ArtifactScopeError),
			datura.WithError(errors.New("missing required field: command")),
		)
	}

	// Create a buffer to hold the output
	var stdoutBuf, stderrBuf bytes.Buffer

	// Execute the command via the runner
	if err := runner.ExecuteCommand(ctx, command, &stdoutBuf, &stderrBuf); err != nil {
		return datura.New(
			datura.WithRole(datura.ArtifactRoleTool),
			datura.WithScope(datura.ArtifactScopeError),
			datura.WithError(err),
		)
	}

	output := stdoutBuf.String()

	if output == "" && stderrBuf.Len() > 0 {
		output = stderrBuf.String()
	}

	return datura.New(
		datura.WithRole(datura.ArtifactRoleTool),
		datura.WithScope(datura.ArtifactScopeResult),
		datura.WithPayload([]byte(output)),
	)
}

func (tool *EnvironmentCommandTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	builder := environment.NewBuilder()

	if builder == nil {
		return mcp.NewToolResultText("Error: Failed to create environment builder"), nil
	}

	runner := environment.NewRunner(builder.Runtime)

	if runner == nil {
		return mcp.NewToolResultText(
			"Error: Failed to create environment runner",
		), nil
	}

	var command string

	for name, value := range req.Params.Arguments {
		if name == "command" {
			command, _ = value.(string)
			break
		}
	}

	if command == "" {
		return mcp.NewToolResultText(
			"Error: Missing required parameter 'command'",
		), nil
	}

	var stdoutBuf, stderrBuf bytes.Buffer

	if err := runner.ExecuteCommand(ctx, command, &stdoutBuf, &stderrBuf); err != nil {
		return mcp.NewToolResultText(
			fmt.Sprintf("Error executing command: %s", err.Error()),
		), nil
	}

	output := stdoutBuf.String()

	if output == "" && stderrBuf.Len() > 0 {
		output = stderrBuf.String()
	}

	if output == "" {
		output = "Command executed successfully with no output"
	}

	return mcp.NewToolResultText(output), nil
}

func (tool *EnvironmentCommandTool) ID() string {
	return "environment_command"
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
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	builder := environment.NewBuilder()

	if builder == nil {
		return nil
	}

	runner := environment.NewRunner(builder.Runtime)

	if runner == nil {
		return nil
	}

	input := datura.GetMetaValue[string](artifact, "input")

	if input == "" {
		return datura.New(
			datura.WithRole(datura.ArtifactRoleTool),
			datura.WithScope(datura.ArtifactScopeError),
			datura.WithError(errors.New("missing required field: input")),
		)
	}

	if input[len(input)-1] != '\n' {
		input += "\n"
	}

	output, err := runner.SendInput(input)

	if err != nil {
		return datura.New(
			datura.WithRole(datura.ArtifactRoleTool),
			datura.WithScope(datura.ArtifactScopeError),
			datura.WithError(err),
		)
	}

	return datura.New(
		datura.WithRole(datura.ArtifactRoleTool),
		datura.WithScope(datura.ArtifactScopeResult),
		datura.WithPayload([]byte(output)),
	)
}

func (tool *EnvironmentInputTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	builder := environment.NewBuilder()

	if builder == nil {
		return mcp.NewToolResultText("Error: Failed to create environment builder"), nil
	}

	runner := environment.NewRunner(builder.Runtime)

	if runner == nil {
		return mcp.NewToolResultText(
			"Error: Failed to create environment runner",
		), nil
	}

	var input string

	for name, value := range req.Params.Arguments {
		if name == "input" {
			input, _ = value.(string)
			break
		}
	}

	if input == "" {
		return mcp.NewToolResultText(
			"Error: Missing required parameter 'input'",
		), nil
	}

	if input[len(input)-1] != '\n' {
		input += "\n"
	}

	output, err := runner.SendInput(input)

	if err != nil {
		return mcp.NewToolResultText(
			fmt.Sprintf("Error sending input: %s", err.Error()),
		), nil
	}

	if output == "" {
		output = "Input accepted"
	}

	return mcp.NewToolResultText(output), nil
}

func (tool *EnvironmentInputTool) ID() string {
	return "environment_input"
}
