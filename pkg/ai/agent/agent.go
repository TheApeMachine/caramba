package agent

import (
	"context"
	"errors"

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
	errnie.Trace("agent.NewAgentServer")

	return &AgentServer{
		agent: agent,
	}
}

/*
AgentToClient converts an Agent to a client capability.
*/
func AgentToClient(agent *AgentBuilder) RPC {
	errnie.Trace("agent.AgentToClient")
	server := NewAgentServer(agent)
	return RPC_ServerToClient(server)
}

/*
Send a message to the agent.
*/
func (srv *AgentServer) Send(ctx context.Context, call RPC_send) error {
	errnie.Trace("agent.Send")

	artifact := datura.New()
	result := errnie.Try(call.AllocResults())

	// Generate response using the provider
	response := srv.agent.AIProvider.Generate(ctx, artifact)
	if response == nil {
		return errnie.Error(errors.New("failed to generate response"))
	}

	// Set the response in the result
	return result.SetOut(*response.Artifact)
}
