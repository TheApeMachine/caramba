package service

import (
	"bytes"
	"context"
	"io"
	"net/http"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/memory"
	"github.com/theapemachine/caramba/pkg/tools"
)

type MCP struct {
	stdio *server.MCPServer
	sse   *server.SSEServer
	tools map[string]io.ReadWriteCloser
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
		tools: map[string]io.ReadWriteCloser{
			"memory": tools.NewMemoryTool(
				memory.NewQdrant(),
				memory.NewNeo4j(),
			),
			"ai":      ai.NewAgent(),
			"editor":  tools.NewEditorTool(),
			"github":  tools.NewGithub(),
			"azure":   tools.NewAzure(),
			"trengo":  tools.NewTrengo(),
			"browser": tools.NewBrowser(),
		},
	}
}

func (service *MCP) Start() error {
	errnie.Debug("MCP.Start")

	service.stdio.AddTool(
		service.tools["memory"].(*tools.MemoryTool).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.memory.tool", req)
			return service.runTool(service.tools["memory"], &req, datura.ArtifactRoleMemoryTool)
		},
	)

	service.stdio.AddTool(
		service.tools["ai"].(*ai.Agent).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.agent.tool", req)
			return service.runTool(service.tools["ai"], &req, datura.ArtifactRoleAgentTool)
		},
	)

	service.stdio.AddTool(
		service.tools["editor"].(*tools.EditorTool).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.editor.tool", req)
			return service.runTool(service.tools["editor"], &req, datura.ArtifactRoleEditorTool)
		},
	)

	service.stdio.AddTool(
		service.tools["github"].(*tools.Github).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.tool", req)
			return service.runTool(service.tools["github"], &req, datura.ArtifactRoleGithubTool)
		},
	)

	service.stdio.AddTool(
		service.tools["azure"].(*tools.Azure).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.azure.tool", req)
			return service.runTool(service.tools["azure"], &req, datura.ArtifactRoleAzureTool)
		},
	)

	service.stdio.AddTool(
		service.tools["trengo"].(*tools.Trengo).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.tool", req)
			return service.runTool(service.tools["trengo"], &req, datura.ArtifactRoleTrengoTool)
		},
	)

	service.stdio.AddTool(
		service.tools["browser"].(*tools.Browser).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.browser.tool", req)
			return service.runTool(service.tools["browser"], &req, datura.ArtifactRoleBrowserTool)
		},
	)

	service.stdio.AddTool(
		service.tools["environment"].(*tools.Environment).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.environment.tool", req)
			return service.runTool(service.tools["environment"], &req, datura.ArtifactRoleEnvironmentTool)
		},
	)

	return nil
}

func (service *MCP) runTool(tool io.ReadWriteCloser, req *mcp.CallToolRequest, role datura.ArtifactRole) (*mcp.CallToolResult, error) {
	errnie.Debug("MCP.runTool")

	options := []datura.ArtifactOption{
		datura.WithRole(role),
	}

	for key, val := range req.Params.Arguments {
		options = append(options, datura.WithMeta(key, val))
	}

	artifact := datura.New(options...)
	buf := bytes.NewBuffer([]byte{})

	if _, err := io.Copy(tool, artifact); err != nil {
		return mcp.NewToolResultText(errnie.Error(err).Error()), nil
	}

	if _, err := io.Copy(buf, tool); err != nil {
		return mcp.NewToolResultText(errnie.Error(err).Error()), nil
	}

	return mcp.NewToolResultText(buf.String()), nil
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
