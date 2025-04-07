package agent

import (
	"context"

	prvdr "github.com/theapemachine/caramba/pkg/ai/provider"
	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
AgentServer implements the RPC interface for the Agent.
*/
type AgentServer struct {
	agent *AgentBuilder
}

/*
NewAgentServer creates a new AgentServer.
*/
func NewAgentServer(agent *AgentBuilder) *AgentServer {
	return &AgentServer{
		agent: agent,
	}
}

/*
AgentToClient converts an Agent to a client capability.
*/
func AgentToClient(agent *AgentBuilder) RPC {
	server := NewAgentServer(agent)
	return RPC_ServerToClient(server)
}

/*
Send a message to the agent.
*/
func (srv *AgentServer) Send(ctx context.Context, call RPC_send) error {
	errnie.Debug("agent.Send")

	artifact := datura.New()

	provider := errnie.Try(srv.agent.Provider())
	result := errnie.Try(call.AllocResults())

	response, release := provider.Client().Generate(
		ctx, func(p prvdr.RPC_generate_Params) error {
			return p.SetContext(*artifact.Artifact)
		},
	)

	defer release()

	out := errnie.Try(response.Struct())
	outArtifact := errnie.Try(out.Out())

	return result.SetOut(outArtifact)
}
