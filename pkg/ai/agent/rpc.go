package agent

import (
	context "context"

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
	errnie.Trace("agent.NewAgentServer")

	return &AgentServer{
		agent: agent,
	}
}

/*
AgentToClient converts an Agent to a client capability.
*/
func AgentToClient(aiAgent *Agent) RPC {
	errnie.Trace("agent.AgentToClient")
	return RPC_ServerToClient(NewAgentServer(aiAgent))
}

/*
Send a message to the agent.
*/
func (srv *AgentServer) Send(ctx context.Context, call RPC_send) error {
	errnie.Trace("agent.Send")

	artifact := errnie.Try(call.Args().Artifact())
	result := errnie.Try(call.AllocResults())

	// client := provider.ProviderToClient(
	// 	errnie.Try(srv.agent.Provider()),
	// )

	// // Generate response using the provider
	// future, release := client.Generate(ctx, func(
	// 	params provider.RPC_generate_Params,
	// ) error {
	// 	return params.SetArtifact(artifact)
	// })

	// defer release()

	// // Wait for the response
	// response, err := future.Struct()

	// if errnie.Error(err) != nil {
	// 	return errnie.Error(err)
	// }

	// Set the response in the result
	return result.SetOut(artifact)
}
