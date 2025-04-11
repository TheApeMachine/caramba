package agent

import (
	"context"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/catalog"
	"github.com/theapemachine/caramba/pkg/tools"
)

type AgentTool struct {
	mcp.Tool
	Tools []tools.Tool
}

// NewAgentTool creates a new agent tool instance.
func NewAgentTool() *AgentTool {
	create := NewCreateAgentTool()

	return &AgentTool{
		Tools: []tools.Tool{
			{
				Tool: create.Tool,
				Use:  create.Use,
			},
		},
	}
}

type CreateAgentTool struct {
	mcp.Tool
}

func NewCreateAgentTool() *CreateAgentTool {
	return &CreateAgentTool{
		Tool: mcp.NewTool(
			"create_agent",
			mcp.WithDescription("A tool which can create an agent."),
			mcp.WithString(
				"name",
				mcp.Description("The name of the agent."),
				mcp.Required(),
			),
			mcp.WithString(
				"description",
				mcp.Description("The description of the agent."),
				mcp.Required(),
			),
			mcp.WithString(
				"system_prompt",
				mcp.Description("The system prompt of the agent."),
				mcp.Required(),
			),
			mcp.WithString(
				"model",
				mcp.Description("The model of the agent."),
				mcp.Required(),
			),
			mcp.WithString(
				"temperature",
				mcp.Description("The temperature of the agent."),
				mcp.Required(),
			),
			mcp.WithString(
				"max_tokens",
				mcp.Description("The maximum number of tokens the agent can use."),
				mcp.Required(),
			),
			mcp.WithString(
				"top_p",
				mcp.Description("The top p of the agent."),
				mcp.Required(),
			),
			mcp.WithString(
				"frequency_penalty",
				mcp.Description("The frequency penalty of the agent."),
				mcp.Required(),
			),
			mcp.WithString(
				"presence_penalty",
				mcp.Description("The presence penalty of the agent."),
				mcp.Required(),
			),
			mcp.WithString(
				"tools",
				mcp.Description("The tools the agent can use."),
				mcp.Required(),
				mcp.Enum(tools.NewRegistry().GetToolNames()...),
			),
		),
	}
}

func (tool *CreateAgentTool) Use(
	ctx context.Context,
	request mcp.CallToolRequest,
) (*mcp.CallToolResult, error) {
	// Extract agent details from the request
	agentName := request.Params.Arguments["name"].(string)
	agentDescription := request.Params.Arguments["description"].(string)
	agentURL := request.Params.Arguments["url"].(string)

	// Create a new agent
	newAgent := &catalog.Agent{
		Name:        agentName,
		Description: agentDescription,
		URL:         agentURL,
	}

	// Add the new agent to the catalog
	catalog := catalog.NewCatalog()
	catalog.AddAgent(newAgent)

	// Return a successful result
	return &mcp.CallToolResult{
		// Populate with appropriate response data
	}, nil
}
