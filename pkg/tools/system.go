package tools

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
)

/* SystemTool provides a base for all system operations */
type SystemTool struct {
	Operations map[string]ToolType
}

/* NewSystemTool creates a new System tool with options */
func NewSystemTool() *SystemTool {

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
	ctx context.Context, artifact *datura.ArtifactBuilder,
) *datura.ArtifactBuilder {
	toolName := datura.GetMetaValue[string](artifact, "tool")
	return tool.Operations[toolName].Use(ctx, artifact)
}

func (tool *SystemTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return tool.Operations[req.Params.Name].UseMCP(ctx, req)
}

/* ToMCP returns all System tool definitions */
func (tool *SystemTool) ToMCP() []ToolType {
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
	ctx context.Context, artifact *datura.ArtifactBuilder,
) *datura.ArtifactBuilder {
	return datura.New(datura.WithArtifact(artifact.Artifact))
}

func (tool *SystemInspectTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SystemOptimizeTool implements a tool for optimizing system performance */
type SystemOptimizeTool struct {
	mcp.Tool
}

/* NewSystemOptimizeTool creates a new tool for system optimization */
func NewSystemOptimizeTool() *SystemOptimizeTool {
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
	ctx context.Context, artifact *datura.ArtifactBuilder,
) *datura.ArtifactBuilder {
	return datura.New(datura.WithArtifact(artifact.Artifact))
}

func (tool *SystemOptimizeTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}

/* SystemMessageTool implements a tool for inter-system communication */
type SystemMessageTool struct {
	mcp.Tool
}

/* NewSystemMessageTool creates a new tool for system messaging */
func NewSystemMessageTool() *SystemMessageTool {
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
	ctx context.Context, artifact *datura.ArtifactBuilder,
) *datura.ArtifactBuilder {
	return datura.New(datura.WithArtifact(artifact.Artifact))
}

func (tool *SystemMessageTool) UseMCP(
	ctx context.Context, req mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	return mcp.NewToolResultText("Hello, world!"), nil
}
