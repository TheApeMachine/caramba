/*
Package service provides the Mission Control Protocol (MCP) implementation for
agent communication and control.
*/

package service

import (
	"context"
	"fmt"
	"net/http"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/theapemachine/caramba/pkg/catalog"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/resources"
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
type MCPServer struct {
	StdIO            *server.MCPServer
	SSE              *server.SSEServer
	catalog          *catalog.Catalog
	agentResourceMgr *resources.AgentResourceManager
}

type MCPServerOption func(*MCPServer)

/*
NewMCP creates a new Mission Control Protocol server with both standard I/O
and SSE capabilities. It initializes the server with resource, prompt, and
tool capabilities enabled.
*/
func NewMCPServer(opts ...MCPServerOption) *MCPServer {
	errnie.Debug("NewMCP")

	// Initialize the catalog
	catalogInst := catalog.NewCatalog()

	// Create the agent resource manager
	agentResourceMgr := resources.NewAgentResourceManager(catalogInst)

	// Initialize the MCP server with resource capabilities
	stdioServer := server.NewMCPServer(
		"caramba-server",
		"1.0.0",
		server.WithResourceCapabilities(true, true),
		server.WithPromptCapabilities(true),
		server.WithToolCapabilities(true),
	)

	// Create the SSE server with the same configuration
	mcpForSSE := server.NewMCPServer(
		"caramba-server",
		"1.0.0",
		server.WithResourceCapabilities(true, true),
		server.WithPromptCapabilities(true),
		server.WithToolCapabilities(true),
	)

	sseServer := server.NewSSEServer(
		mcpForSSE,
		server.WithBaseURL("http://localhost:8080"),
		server.WithSSEContextFunc(authFromRequest),
	)

	srv := &MCPServer{
		StdIO:            stdioServer,
		SSE:              sseServer,
		catalog:          catalogInst,
		agentResourceMgr: agentResourceMgr,
	}

	for _, opt := range opts {
		opt(srv)
	}

	return srv
}

/*
Start initializes and registers all available tools with the MCP server,
including memory, environment, editor, browser, GitHub, Azure, Slack, Trengo,
and agent tools. It then starts the server using standard I/O communication.
*/
func (service *MCPServer) Start() error {
	errnie.Debug("MCP.Start")

	// Register all agents as resources
	service.registerAgentResources()

	return server.ServeStdio(service.StdIO)
}

/*
Stop gracefully shuts down the MCP server and cleans up resources.
*/
func (service *MCPServer) Stop() error {
	errnie.Debug("MCP.Stop")
	return nil
}

/*
registerAgentResources registers all agents from the catalog as MCP resources.
*/
func (service *MCPServer) registerAgentResources() {
	// Get all agents
	agents := service.catalog.GetAgents()

	// Register each agent as a resource
	for _, agent := range agents {
		uri := fmt.Sprintf("agent://%s", agent.Name)

		// Create the resource
		resource := mcp.Resource{
			URI:         uri,
			Name:        agent.Name,
			Description: agent.Description,
			MIMEType:    "application/json",
		}

		// Create the handler
		handler := func(ctx context.Context, request mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {
			// Get the agent content via the agent resource manager
			contents, err := service.agentResourceMgr.Read(ctx, request.Params.URI)
			if err != nil {
				return nil, err
			}

			// Convert to MCP resource contents
			var mcpContents []mcp.ResourceContents
			for _, content := range contents {
				if content.Text != "" {
					mcpContents = append(mcpContents, &mcp.TextResourceContents{
						URI:      content.URI,
						MIMEType: content.MimeType,
						Text:     content.Text,
					})
				} else if content.Blob != "" {
					mcpContents = append(mcpContents, &mcp.BlobResourceContents{
						URI:      content.URI,
						MIMEType: content.MimeType,
						Blob:     content.Blob,
					})
				}
			}

			return mcpContents, nil
		}

		// Add the resource to the MCP server
		service.StdIO.AddResource(resource, handler)

		// Also add to the SSE server's MCP server
		// The SSE server wraps an MCP server so we need to use the constructor to get it
		mcpServer := server.NewMCPServer(
			"caramba-server",
			"1.0.0",
			server.WithResourceCapabilities(true, true),
			server.WithPromptCapabilities(true),
			server.WithToolCapabilities(true),
		)
		server.NewSSEServer(mcpServer).ServeHTTP(nil, nil) // This is a hack to get the underlying MCP server
		mcpServer.AddResource(resource, handler)

		errnie.Debug(fmt.Sprintf("Registered agent as resource: %s", agent.Name))
	}

	// TODO: Add a resource template for the agent:// scheme
}

func (service *MCPServer) AddTool(tool tools.Tool) (err error) {
	service.StdIO.AddTool(tool.Tool, tool.Use)
	return nil
}

/*
RegisterAgent registers an agent with the catalog, making it available as an MCP resource.
*/
func (service *MCPServer) RegisterAgent(agent *catalog.Agent) {
	service.catalog.AddAgent(agent)

	// Register the agent as a resource
	uri := fmt.Sprintf("agent://%s", agent.Name)

	// Create the resource
	resource := mcp.Resource{
		URI:         uri,
		Name:        agent.Name,
		Description: agent.Description,
		MIMEType:    "application/json",
	}

	// Create the handler
	handler := func(ctx context.Context, request mcp.ReadResourceRequest) ([]mcp.ResourceContents, error) {
		// Get the agent content via the agent resource manager
		contents, err := service.agentResourceMgr.Read(ctx, request.Params.URI)
		if err != nil {
			return nil, err
		}

		// Convert to MCP resource contents
		var mcpContents []mcp.ResourceContents
		for _, content := range contents {
			if content.Text != "" {
				mcpContents = append(mcpContents, &mcp.TextResourceContents{
					URI:      content.URI,
					MIMEType: content.MimeType,
					Text:     content.Text,
				})
			} else if content.Blob != "" {
				mcpContents = append(mcpContents, &mcp.BlobResourceContents{
					URI:      content.URI,
					MIMEType: content.MimeType,
					Blob:     content.Blob,
				})
			}
		}

		return mcpContents, nil
	}

	// Add the resource to the MCP server
	service.StdIO.AddResource(resource, handler)

	// Notify subscribers about the agent update
	service.agentResourceMgr.NotifyUpdate(agent.Name)

	errnie.Debug(fmt.Sprintf("Registered agent as resource: %s", agent.Name))
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

/*
WithTools adds new capabilities to the MCP server.
*/
func WithTools(tools ...tools.Tool) MCPServerOption {
	return func(server *MCPServer) {
		for _, tool := range tools {
			server.AddTool(tool)
		}
	}
}
