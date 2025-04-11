/*
Package service provides the Mission Control Protocol (MCP) implementation for
agent communication and control.
*/

package service

import (
	"context"
	"net/http"

	"github.com/mark3labs/mcp-go/server"
	"github.com/theapemachine/caramba/pkg/agent"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tools"
)

/*
MCP implements the Mission Control Protocol server, providing both standard I/O
and Server-Sent Events (SSE) interfaces for agent communication. It supports
resource management, prompt handling, and tool capabilities.

Example:

	mcp := NewMCP()
	if err := mcp.Start(); err != nil {
	    log.Fatal(err)
	}
	defer mcp.Stop()
*/
type MCP struct {
	StdIO *server.MCPServer
	SSE   *server.SSEServer
}

/*
NewMCP creates a new Mission Control Protocol server with both standard I/O
and SSE capabilities. It initializes the server with resource, prompt, and
tool capabilities enabled.
*/
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

/*
Start initializes and registers all available tools with the MCP server,
including memory, environment, editor, browser, GitHub, Azure, Slack, Trengo,
and agent tools. It then starts the server using standard I/O communication.
*/
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

/*
Stop gracefully shuts down the MCP server and cleans up resources.
*/
func (service *MCP) Stop() error {
	errnie.Debug("MCP.Stop")
	return nil
}

/*
authKey is a context key type for storing authentication information.
*/
type authKey struct{}

/*
authFromRequest extracts the Authorization header from an HTTP request and
stores it in the request context for later use.
*/
func authFromRequest(ctx context.Context, r *http.Request) context.Context {
	return withAuthKey(ctx, r.Header.Get("Authorization"))
}

/*
withAuthKey stores the authentication key in the context for use throughout

	the request lifecycle.
*/
func withAuthKey(ctx context.Context, auth string) context.Context {
	return context.WithValue(ctx, authKey{}, auth)
}
