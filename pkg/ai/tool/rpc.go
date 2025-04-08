package tool

import (
	context "context"

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

	result := errnie.Try(call.AllocResults())

	return result.SetOut(srv.tool.MCPTool.Use(
		ctx, errnie.Try(call.Args().Artifact()),
	))
}

func ToolToClient(tool *ToolBuilder) RPC {
	errnie.Trace("tool.ToolToClient")
	return RPC_ServerToClient(NewToolRPCServer(tool))
}
