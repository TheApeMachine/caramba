package tool

import (
	context "context"

	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ToolRPCServer struct {
	tool *ToolBuilder
}

func NewToolRPCServer(tool *ToolBuilder) *ToolRPCServer {
	errnie.Trace("tool.NewToolRPCServer")

	return &ToolRPCServer{tool: tool}
}

func (srv *ToolRPCServer) Use(
	ctx context.Context,
	call RPC_use,
) (err error) {
	errnie.Trace("tool.Use RPC", "tool_name", srv.tool.Name)

	artifact := errnie.Try(call.Args().Artifact())
	result := errnie.Try(call.AllocResults())

	artifactBuilder := datura.New(
		datura.WithArtifact(&artifact),
	)

	responseBuilder := srv.tool.MCPTool.Use(ctx, artifactBuilder)

	return result.SetOut(*responseBuilder.Artifact)
}

func ToolToClient(tool *ToolBuilder) RPC {
	errnie.Trace("tool.ToolToClient")

	server := NewToolRPCServer(tool)
	return RPC_ServerToClient(server)
}
