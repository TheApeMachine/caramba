package service

import (
	"context"
	"net/http"

	"github.com/mark3labs/mcp-go/server"
)

type MCP struct {
	stdio *server.MCPServer
	sse   *server.SSEServer
}

func NewMCP() *MCP {
	return &MCP{
		stdio: server.NewMCPServer(
			"example-server",
			"1.0.0",
			server.WithResourceCapabilities(true, true),
			server.WithPromptCapabilities(true),
			server.WithToolCapabilities(true),
		),
		sse: server.NewSSEServer(
			server.NewMCPServer(
				"example-server",
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
	return nil
}

func (service *MCP) Stop() error {
	return nil
}

type authKey struct{}

func authFromRequest(ctx context.Context, r *http.Request) context.Context {
	return withAuthKey(ctx, r.Header.Get("Authorization"))
}

func withAuthKey(ctx context.Context, auth string) context.Context {
	return context.WithValue(ctx, authKey{}, auth)
}
