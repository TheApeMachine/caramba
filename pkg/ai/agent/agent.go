package agent

import (
	"context"

	aictx "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
AgentServer implements the RPC interface for the Agent.
*/
type AgentServer struct {
	agent *Agent
}

/*
NewAgentServer creates a new AgentServer.
*/
func NewAgentServer(agent *Agent) *AgentServer {
	return &AgentServer{
		agent: agent,
	}
}

/*
AgentToClient converts an Agent to a client capability.
*/
func AgentToClient(agent *Agent) RPC {
	server := NewAgentServer(agent)
	return RPC_ServerToClient(server)
}

/*
Send a message to the agent.
*/
func (srv *AgentServer) Send(ctx context.Context, call RPC_send) error {
	msg := errnie.Try(call.Args().Message_())
	agentCtx := errnie.Try(srv.agent.Context())

	future, release := aictx.ContextToClient(
		&agentCtx).Add(ctx, func(p aictx.RPC_add_Params) error {
		return p.SetContext(msg)
	})

	defer release()

	_, err := future.Struct()
	return err
}
