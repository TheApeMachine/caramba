package service

import (
	"context"
	"net/http"

	"github.com/mark3labs/mcp-go/server"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools"
)

type MCP struct {
	stdio *server.MCPServer
	sse   *server.SSEServer
}

func NewMCP() *MCP {
	errnie.Debug("NewMCP")

	return &MCP{
		stdio: server.NewMCPServer(
			"caramba-server",
			"1.0.0",
			server.WithResourceCapabilities(true, true),
			server.WithPromptCapabilities(true),
			server.WithToolCapabilities(true),
		),
		sse: server.NewSSEServer(
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

	for _, tool := range tools.NewSystemTool().ToMCP() {
		service.stdio.AddTool(tool.Tool, tool.UseMCP)
	}

	for _, tool := range tools.NewMemoryTool(datura.New()).ToMCP() {
		service.stdio.AddTool(tool.Tool, tool.UseMCP)
	}

	for _, tool := range tools.NewEnvironmentTool().ToMCP() {
		service.stdio.AddTool(tool.Tool, tool.UseMCP)
	}

	for _, tool := range tools.NewEditorTool(datura.New()).ToMCP() {
		service.stdio.AddTool(tool.Tool, tool.UseMCP)
	}

	for _, tool := range tools.NewBrowserTool(datura.New()).ToMCP() {
		service.stdio.AddTool(tool.Tool, tool.UseMCP)
	}

	for _, tool := range tools.NewGithubTool(datura.New()).ToMCP() {
		service.stdio.AddTool(tool.Tool, tool.UseMCP)
	}

	for _, tool := range tools.NewAzureTool(datura.New()).ToMCP() {
		service.stdio.AddTool(tool.Tool, tool.UseMCP)
	}

	for _, tool := range tools.NewSlackTool(datura.New()).ToMCP() {
		service.stdio.AddTool(tool.Tool, tool.UseMCP)
	}

	for _, tool := range tools.NewTrengoTool(datura.New()).ToMCP() {
		service.stdio.AddTool(tool.Tool, tool.UseMCP)
	}

	return nil
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
