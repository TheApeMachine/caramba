package service

import (
	"context"
	"net/http"

	"github.com/mark3labs/mcp-go/server"
	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools"
)

type MCP struct {
	StdIO *server.MCPServer
	SSE   *server.SSEServer
}

func NewMCP() *MCP {
	errnie.Debug("NewMCP")

	return &MCP{
		StdIO: server.NewMCPServer(
			"caramba-server",
			"1.0.0",
			server.WithResourceCapabilities(true, true),
			server.WithPromptCapabilities(true),
			server.WithToolCapabilities(true),
		),
		SSE: server.NewSSEServer(
			server.NewMCPServer(
				"caramba-server",
				"1.0.0",
				server.WithResourceCapabilities(true, true),
				server.WithPromptCapabilities(true),
				server.WithToolCapabilities(true),
			),
			server.WithBaseURL("http://localhost:8080"),
			server.WithSSEContextFunc(authFromRequest),
		),
	}
}

func (service *MCP) Start() error {
	errnie.Debug("MCP.Start")

	for _, tool := range tools.NewMemoryTool().Tools {
		service.StdIO.AddTool(tool.Tool, tool.Use)
	}

	for _, tool := range tools.NewEnvironmentTool().Tools {
		service.StdIO.AddTool(tool.Tool, tool.Use)
	}

	for _, tool := range tools.NewEditorTool().Tools {
		service.StdIO.AddTool(tool.Tool, tool.Use)
	}

	for _, tool := range tools.NewBrowserTool().Tools {
		service.StdIO.AddTool(tool.Tool, tool.Use)
	}

	for _, tool := range tools.NewGithubTool().Tools {
		service.StdIO.AddTool(tool.Tool, tool.Use)
	}

	for _, tool := range tools.NewAzureTool().Tools {
		service.StdIO.AddTool(tool.Tool, tool.Use)
	}

	for _, tool := range tools.NewSlackTool().Tools {
		service.StdIO.AddTool(tool.Tool, tool.Use)
	}

	for _, tool := range tools.NewTrengoTool().Tools {
		service.StdIO.AddTool(tool.Tool, tool.Use)
	}

	for _, tool := range agent.NewAgentTool().Tools {
		service.StdIO.AddTool(tool.Tool, tool.Use)
	}

	return server.ServeStdio(service.StdIO)
}

func (service *MCP) Stop() error {
	errnie.Debug("MCP.Stop")
	return nil
}

type authKey struct{}

func authFromRequest(ctx context.Context, r *http.Request) context.Context {
	return withAuthKey(ctx, r.Header.Get("Authorization"))
}

func withAuthKey(ctx context.Context, auth string) context.Context {
	return context.WithValue(ctx, authKey{}, auth)
}
