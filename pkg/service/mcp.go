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
				map[string]io.ReadWriteCloser{
					"qdrant": memory.NewQdrant(),
					"neo4j":  memory.NewNeo4j(),
				},
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
	service.stdio.AddTool(
		service.tools["memory"].(*tools.MemoryTool).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.memory.tool", req)

			artifact := datura.New(
				datura.WithMeta("tool", "memory"),
			)

			return service.runTool(service.tools["memory"], artifact)
		},
	)

	service.stdio.AddTool(
		service.tools["ai"].(*ai.Agent).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.agent.tool", req)

			artifact := datura.New(
				datura.WithMeta("tool", "ai"),
			)

			return service.runTool(service.tools["ai"], artifact)
		},
	)

	service.stdio.AddTool(
		service.tools["editor"].(*tools.EditorTool).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.editor.tool", req)

			artifact := datura.New(
				datura.WithMeta("tool", "editor"),
			)

			return service.runTool(service.tools["editor"], artifact)
		},
	)

	service.stdio.AddTool(
		service.tools["github"].(*tools.Github).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.github.tool", req)

			artifact := datura.New(
				datura.WithMeta("tool", "github"),
			)

			return service.runTool(service.tools["github"], artifact)
		},
	)

	service.stdio.AddTool(
		service.tools["azure"].(*tools.Azure).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.azure.tool", req)

			artifact := datura.New(
				datura.WithMeta("tool", "azure"),
			)

			return service.runTool(service.tools["azure"], artifact)
		},
	)

	service.stdio.AddTool(
		service.tools["trengo"].(*tools.Trengo).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.trengo.tool", req)

			artifact := datura.New(
				datura.WithMeta("tool", "trengo"),
			)

			return service.runTool(service.tools["trengo"], artifact)
		},
	)

	service.stdio.AddTool(
		service.tools["browser"].(*tools.Browser).Schema.ToMCP(),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			errnie.Debug("mcp.browser.tool", req)

			artifact := datura.New(
				datura.WithMeta("tool", "browser"),
			)

			return service.runTool(service.tools["browser"], artifact)
		},
	)

	return nil
}

func (service *MCP) runTool(tool io.ReadWriteCloser, artifact *datura.Artifact) (*mcp.CallToolResult, error) {
	buf := bytes.NewBuffer([]byte{})

	if _, err := io.Copy(tool, artifact); err != nil {
		return mcp.NewToolResultError(errnie.Error(err).Error()), nil
	}

	if _, err := io.Copy(buf, tool); err != nil {
		return mcp.NewToolResultError(errnie.Error(err).Error()), nil
	}

	return mcp.NewToolResultText(buf.String()), nil
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
