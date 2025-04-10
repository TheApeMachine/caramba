package tools

import (
	"context"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/* SystemTool provides a base for all system operations */
type SystemTool struct {
	Operations map[string]ToolType
}

/* NewSystemTool creates a new System tool with options */
func NewSystemTool() *SystemTool {
	errnie.Trace("tools.NewSystemTool")

	inspect := NewSystemInspectTool()
	optimize := NewSystemOptimizeTool()
	message := NewSystemMessageTool()

	return &SystemTool{
		Operations: map[string]ToolType{
			"system_inspect":  {inspect.Tool, inspect.Use, inspect.UseMCP},
			"system_optimize": {optimize.Tool, optimize.Use, optimize.UseMCP},
			"system_message":  {message.Tool, message.Use, message.UseMCP},
		},
	}
}

func (tool *SystemTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("tools.SystemTool.Use")

	toolName := datura.GetMetaValue[string](artifact, "tool")
	return tool.Operations[toolName].Use(ctx, artifact)
}

func (tool *SystemTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	errnie.Trace("tools.SystemTool.UseMCP")

	return tool.Operations[req.Params.Name].UseMCP(ctx, req)
}

/* ToMCP returns all System tool definitions */
func (tool *SystemTool) ToMCP() []ToolType {
	errnie.Trace("tools.SystemTool.ToMCP")

	tools := make([]ToolType, 0)

	for _, tool := range tool.Operations {
		tools = append(tools, tool)
	}

	return tools
}

/* SystemInspectTool implements a tool for inspecting system state */
type SystemInspectTool struct {
	mcp.Tool
}

/* NewSystemInspectTool creates a new tool for system inspection */
func NewSystemInspectTool() *SystemInspectTool {
	errnie.Trace("tools.NewSystemInspectTool")

	return &SystemInspectTool{
		Tool: mcp.NewTool(
			"system_inspect",
			mcp.WithDescription("A tool for inspecting the system."),
			mcp.WithString(
				"scope",
				mcp.Description("The scope of the inspection."),
				mcp.Enum("agents", "topics"),
				mcp.Required(),
			),
		),
	}
}

/* Use executes the system inspection operation and returns the results */
func (tool *SystemInspectTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("tools.SystemInspectTool.Use")

	return datura.New(
		datura.WithArtifact(artifact),
	)
}

func (tool *SystemInspectTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	errnie.Trace("tools.SystemInspectTool.UseMCP")

	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SystemOptimizeTool implements a tool for optimizing system performance */
type SystemOptimizeTool struct {
	mcp.Tool
}

/* NewSystemOptimizeTool creates a new tool for system optimization */
func NewSystemOptimizeTool() *SystemOptimizeTool {
	errnie.Trace("tools.NewSystemOptimizeTool")

	return &SystemOptimizeTool{
		Tool: mcp.NewTool(
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
		),
	}
}

/* Use executes the system optimization operation and returns the results */
func (tool *SystemOptimizeTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("tools.SystemOptimizeTool.Use")

	return datura.New(
		datura.WithArtifact(artifact),
	)
}

func (tool *SystemOptimizeTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	errnie.Trace("tools.SystemOptimizeTool.UseMCP")

	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SystemMessageTool implements a tool for inter-system communication */
type SystemMessageTool struct {
	mcp.Tool
}

/* NewSystemMessageTool creates a new tool for system messaging */
func NewSystemMessageTool() *SystemMessageTool {
	errnie.Trace("tools.NewSystemMessageTool")

	return &SystemMessageTool{
		Tool: mcp.NewTool(
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
		),
	}
}

/* Use executes the system messaging operation and returns the results */
func (tool *SystemMessageTool) Use(
	ctx context.Context, artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("tools.SystemMessageTool.Use")

	// Extract message details from the artifact
	targetAgent := datura.GetMetaValue[string](artifact, "to")
	message := datura.GetMetaValue[string](artifact, "message")
	fromAgent := datura.GetMetaValue[string](artifact, "origin")

	if targetAgent == "" || message == "" {
		return datura.New(
			datura.WithArtifact(artifact),
			datura.WithMetadata(map[string]any{
				"error": "missing required fields: to and/or message",
			}),
		)
	}

	// Create a private topic for the target agent
	privateTopic := fmt.Sprintf("%s-private", targetAgent)

	// Create a new artifact for the private channel and it's automatically registered
	datura.New(
		datura.WithPayload([]byte(message)),
		datura.WithMetadata(map[string]any{
			"from":    fromAgent,
			"to":      targetAgent,
			"message": message,
			"topic":   privateTopic,
		}),
	)

	// The message will be delivered via the hub's topic system
	// This is handled by the agent service

	// Create response artifact
	return datura.New(
		datura.WithArtifact(artifact),
		datura.WithMetadata(map[string]any{
			"result":  "Message sent successfully",
			"to":      targetAgent,
			"message": message,
		}),
	)
}

func (tool *SystemMessageTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	errnie.Trace("tools.SystemMessageTool.UseMCP")

	// Extract parameters
	to := ""
	message := ""
	var fromAgent string = "system_message"

	for name, value := range req.Params.Arguments {
		if name == "to" {
			to, _ = value.(string)
		}
		if name == "message" {
			message, _ = value.(string)
		}
	}

	if to == "" || message == "" {
		return mcp.NewToolResultText("Error: Missing required parameters 'to' and/or 'message'"), nil
	}

	// Create a private topic for the target agent
	privateTopic := fmt.Sprintf("%s-private", to)

	// Create a message artifact for the private topic
	datura.New(
		datura.WithPayload([]byte(message)),
		datura.WithMetadata(map[string]any{
			"from":    fromAgent,
			"to":      to,
			"message": message,
			"topic":   privateTopic,
		}),
	)

	// The message will be delivered via the hub's topic system

	return mcp.NewToolResultText(fmt.Sprintf("Message sent to %s: %s", to, message)), nil
}
