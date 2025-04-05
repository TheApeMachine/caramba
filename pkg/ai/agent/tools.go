package agent

import (
	context "context"
	"encoding/json"
	"fmt"

	"capnproto.org/go/capnp/v3"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/tools"
)

// AsTool implements the tool interface for the agent
func (srv *AgentServer) AsTool(ctx context.Context, call AgentRPC_asTool) error {
	name, err := call.Args().Name()
	if err != nil {
		return err
	}

	args, err := call.Args().Args()
	if err != nil {
		return err
	}

	// Create a message for tool execution
	msg := datura.New(
		datura.WithRole(datura.ArtifactRoleTool),
		datura.WithScope(datura.ArtifactScopeGeneration),
		datura.WithMetadata(map[string]any{
			"tool": name,
			"args": args,
		}),
	)

	// Process the tool request
	if err := srv.handleMessage(ctx, msg); err != nil {
		return err
	}

	// Set response
	results, err := call.AllocResults()
	if err != nil {
		return err
	}

	return results.SetResult("Tool execution completed")
}

type ToolsServer struct {
	agent *Agent
}

func NewToolsServer(agent *Agent) *ToolsServer {
	return &ToolsServer{agent: agent}
}

func (srv *ToolsServer) Add(ctx context.Context, call Tools_add) error {
	tl, err := srv.agent.Tools()
	if err != nil {
		return err
	}

	ntl, err := capnp.NewTextList(srv.agent.Segment(), int32(tl.Len()+1))
	if err != nil {
		return err
	}

	// Copy existing tools
	for i := 0; i < tl.Len(); i++ {
		tool, err := tl.At(i)
		if err != nil {
			return err
		}

		if err := ntl.Set(i, tool); err != nil {
			return err
		}
	}

	// Add new tool
	name, err := call.Args().Name()
	if err != nil {
		return err
	}

	if err := ntl.Set(tl.Len(), name); err != nil {
		return err
	}

	// Update the agent's tools list
	return srv.agent.SetTools(ntl)
}

func (srv *ToolsServer) Use(ctx context.Context, call Tools_use) error {
	// Get the tool name and arguments from the call
	name, err := call.Args().Name()
	if err != nil {
		return err
	}

	arguments, err := call.Args().Arguments()

	if err != nil {
		return err
	}

	args := map[string]any{}

	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return err
	}

	// Get the tool implementation
	tools := srv.getTool(name)

	if len(tools) == 0 {
		return fmt.Errorf("tool %s not found", name)
	}

	// Execute the tool
	req := mcp.CallToolRequest{
		Params: struct {
			Name      string                 `json:"name"`
			Arguments map[string]interface{} `json:"arguments,omitempty"`
			Meta      *struct {
				ProgressToken mcp.ProgressToken `json:"progressToken,omitempty"`
			} `json:"_meta,omitempty"`
		}{
			Name:      name,
			Arguments: args,
		},
	}

	result, err := tools[0].Use(ctx, req)
	if err != nil {
		return err
	}

	// Set the result
	results, err := call.AllocResults()
	if err != nil {
		return err
	}

	if result != nil && len(result.Content) > 0 {
		// Try to convert content to TextContent
		if textContent, ok := mcp.AsTextContent(result.Content[0]); ok {
			return results.SetResult(textContent.Text)
		}
	}
	return results.SetResult("")
}

func (srv *ToolsServer) getTool(name string) []tools.ToolType {
	switch name {
	case "system":
		return tools.NewSystemTool().ToMCP()
	case "browser":
		return tools.NewBrowserTool().ToMCP()
	case "environment":
		return tools.NewEnvironmentTool().ToMCP()
	case "azure":
		return tools.NewAzureTool().ToMCP()
	case "editor":
		return tools.NewEditorTool().ToMCP()
	case "github":
		return tools.NewGithubTool().ToMCP()
	case "memory":
		return tools.NewMemoryTool().ToMCP()
	case "slack":
		return tools.NewSlackTool().ToMCP()
	case "trengo":
		return tools.NewTrengoTool().ToMCP()
	}

	return nil
}
