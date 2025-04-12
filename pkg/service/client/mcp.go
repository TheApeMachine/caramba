package client

import (
	"context"
	"fmt"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

/*
MCPClient is a Model Context Protocol client.

It is used to facilitate a standardized path to integrate MCP-based tools
into the Caramba ecosystem. It provides a streaming interface to MCP operations
and implements io.ReadWriteCloser for streaming data processing.
*/
type MCPClient struct {
	client client.MCPClient
	ctx    context.Context
}

/*
NewMCP creates a new MCP client instance.

It initializes an MCP client based on configuration and sets up a buffered stream
for processing MCP operations. The client can be either SSE-based or stdio-based.
*/
func NewMCPClient() *MCPClient {
	// Get configuration
	baseURL := tweaker.Get("tools.mcp.baseURL", "http://localhost:3000")
	clientType := tweaker.Get("tools.mcp.clientType", "sse") // "sse" or "stdio"

	var mcpClient client.MCPClient
	var err error

	// Create the appropriate client type based on configuration
	if clientType == "sse" {
		mcpClient, err = client.NewSSEMCPClient(baseURL)
	} else {
		// For stdio client, we need command, env, and args
		command := tweaker.Get("tools.mcp.command", "")
		env := tweaker.GetStringSlice("tools.mcp.env")
		args := tweaker.GetStringSlice("tools.mcp.args")

		mcpClient, err = client.NewStdioMCPClient(command, env, args...)
	}

	if err != nil {
		errnie.Error(fmt.Errorf("failed to create MCP client: %w", err))
		return nil
	}

	ctx := context.Background()

	// Register notification handler
	mcpClient.OnNotification(func(notification mcp.JSONRPCNotification) {
		errnie.Debug(fmt.Sprintf("MCP notification received: %v", notification))
	})

	// For SSE client, start the connection
	if sseClient, ok := mcpClient.(*client.SSEMCPClient); ok {
		if err := sseClient.Start(ctx); err != nil {
			errnie.Error(fmt.Errorf("failed to start SSE client: %w", err))
			return nil
		}
	}

	return &MCPClient{
		client: mcpClient,
		ctx:    ctx,
	}
}

/*
CallTool calls a tool via the MCP client.
*/
func (m *MCPClient) CallTool(request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return m.client.CallTool(m.ctx, request)
}

/*
Complete sends a completion request to the MCP client.
*/
func (m *MCPClient) Complete(request mcp.CompleteRequest) (*mcp.CompleteResult, error) {
	return m.client.Complete(m.ctx, request)
}

/*
Initialize initializes the MCP client session.
*/
func (m *MCPClient) Initialize(request mcp.InitializeRequest) (*mcp.InitializeResult, error) {
	return m.client.Initialize(m.ctx, request)
}

/*
ListTools requests the available tools from the MCP client.
*/
func (m *MCPClient) ListTools(request mcp.ListToolsRequest) (*mcp.ListToolsResult, error) {
	return m.client.ListTools(m.ctx, request)
}

/*
ListPrompts requests the available prompts from the MCP client.
*/
func (m *MCPClient) ListPrompts(request mcp.ListPromptsRequest) (*mcp.ListPromptsResult, error) {
	return m.client.ListPrompts(m.ctx, request)
}

/*
GetPrompt retrieves a specific prompt from the MCP client.
*/
func (m *MCPClient) GetPrompt(request mcp.GetPromptRequest) (*mcp.GetPromptResult, error) {
	return m.client.GetPrompt(m.ctx, request)
}
