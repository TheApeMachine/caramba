package client

import (
	"context"
	"fmt"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
MCPClient is a Model Context Protocol client.

It is used to facilitate a standardized path to integrate MCP-based tools
into the Caramba ecosystem. It provides a streaming interface to MCP operations
and implements io.ReadWriteCloser for streaming data processing.
*/
type MCPClient struct {
	StdIO client.MCPClient
	SSE   client.MCPClient
}

type MCPClientOption func(*MCPClient)

/*
NewMCP creates a new MCP client instance.

It initializes an MCP client based on configuration and sets up a buffered stream
for processing MCP operations. The client can be either SSE-based or stdio-based.
*/
func NewMCPClient(opts ...MCPClientOption) *MCPClient {
	mcpc := &MCPClient{}

	for _, opt := range opts {
		opt(mcpc)
	}

	return mcpc
}

func (mcpc *MCPClient) Start(ctx context.Context) error {
	mcpc.SSE.OnNotification(func(notification mcp.JSONRPCNotification) {
		errnie.Debug(fmt.Sprintf("MCP notification received: %v", notification))
	})

	return nil
}

func WithStdIO(command string, env []string, args ...string) MCPClientOption {
	return func(mcpc *MCPClient) {
		var err error

		if mcpc.StdIO, err = client.NewStdioMCPClient(
			command, env, args...,
		); err != nil {
			errnie.New(errnie.WithError(err))
			return
		}
	}
}

func WithSSE(baseURL string) MCPClientOption {
	return func(mcpc *MCPClient) {
		var err error

		if mcpc.SSE, err = client.NewSSEMCPClient(baseURL); err != nil {
			errnie.New(errnie.WithError(err))
			return
		}
	}
}

func WithNotificationHandler(
	handler func(notification mcp.JSONRPCNotification),
) MCPClientOption {
	return func(mcpc *MCPClient) {
		mcpc.SSE.OnNotification(handler)
	}
}
