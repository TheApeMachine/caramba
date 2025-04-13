package agents

import (
	"context"
	"strings"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/registry"
)

type List struct {
	mcp.Tool
}

func NewList() *List {
	return &List{
		Tool: mcp.NewTool(
			"list_agents",
			mcp.WithDescription("List the remote agents you have access to."),
		),
	}
}

func (t *List) Use(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	registry := registry.GetAmbient()

	var localAgent agent.Card
	var agents []agent.Card

	if err := registry.Get(ctx, "local_agent", &localAgent); err != nil {
		return mcp.NewToolResultText("error getting local agent"), errnie.New(errnie.WithError(err))
	}

	if err := registry.Get(ctx, "agents", &agents); err != nil {
		return mcp.NewToolResultText("error getting agents"), errnie.New(errnie.WithError(err))
	}

	var result strings.Builder

	for _, agent := range agents {
		result.WriteString(agent.Name + "\n")
		result.WriteString(agent.Description + "\n")
		result.WriteString(agent.URL + "\n")
	}

	return mcp.NewToolResultText(result.String()), nil
}
