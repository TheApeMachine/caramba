package agent

import (
	"context"

	"capnproto.org/go/capnp/v3"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/params"
	"github.com/theapemachine/caramba/pkg/ai/tool"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools"
)

// handleMessage processes incoming messages
func (srv *AgentServer) handleMessage(
	ctx context.Context, msg *datura.ArtifactBuilder,
) error {
	// Check message role and scope
	switch datura.ArtifactRole(msg.Role()) {
	case datura.ArtifactRoleQuestion:
		// Handle questions/requests from other agents
		return srv.handleQuestion(ctx, msg)
	case datura.ArtifactRoleSystem:
		// Handle system commands/tasks from other agents
		return srv.handleCommand(ctx, msg)
	case datura.ArtifactRoleData:
		// Handle data/information from other agents
		return srv.handleData(ctx, msg)
	default:
		// Default message handling
		return srv.processMessage(ctx, datura.GetMetaValue[string](msg, "topic"), msg)
	}
}

// Process handles incoming messages from other agents.
func (srv *AgentServer) Process(ctx context.Context, call AgentRPC_process) error {
	// Get message data
	msgData := call.Args().Message()

	// The message is already a capnp.Message, so we can read the root directly
	artifact, err := datura.ReadRootArtifact(msgData)
	if err != nil {
		return err
	}

	// Queue message for processing
	msg := datura.New()
	msg.Artifact = &artifact
	srv.msgCh <- msg

	// Prepare response
	response := datura.New(
		datura.WithRole(datura.ArtifactRoleAcknowledge),
		datura.WithScope(datura.ArtifactScopeGeneration),
	)

	// Marshal response
	buf, err := response.Message().Marshal()
	if err != nil {
		return err
	}

	// Set response
	results, err := call.AllocResults()
	if err != nil {
		return err
	}
	return results.SetResponse(buf)
}

// handleQuestion processes questions from other agents
func (srv *AgentServer) handleQuestion(
	ctx context.Context,
	msg *datura.ArtifactBuilder,
) error {
	// TODO: Implement question handling based on agent role
	return nil
}

// handleCommand processes commands/tasks from other agents
func (srv *AgentServer) handleCommand(
	ctx context.Context,
	msg *datura.ArtifactBuilder,
) error {
	// Create a new OpenAI provider
	prvdr := provider.NewOpenAIProvider()

	// Get the tools from the agent's tools list
	toolList, err := srv.agent.Tools()
	if err != nil {
		return errnie.Error(err)
	}

	// Convert tools to MCP format
	var mcpTools []tools.ToolType
	for i := 0; i < toolList.Len(); i++ {
		name, err := toolList.At(i)
		if err != nil {
			return errnie.Error(err)
		}

		builder := tool.New(tool.WithName(name))
		if builder != nil {
			mcpTools = append(mcpTools, builder.ToMCP().MCP...)
		}
	}

	// Get the task description from the message payload
	payload, err := msg.DecryptPayload()
	if err != nil {
		return errnie.Error(err)
	}

	// Create a new context for the provider
	_, seg, err := capnp.NewMessage(capnp.SingleSegment(nil))
	if err != nil {
		return errnie.Error(err)
	}

	context, err := aicontext.NewRootContext(seg)
	if err != nil {
		return errnie.Error(err)
	}

	// Add the task description as a user message
	msgs, err := context.NewMessages(1)
	if err != nil {
		return errnie.Error(err)
	}
	msg0 := msgs.At(0)
	if err := msg0.SetRole("user"); err != nil {
		return errnie.Error(err)
	}
	if err := msg0.SetContent(string(payload)); err != nil {
		return errnie.Error(err)
	}

	// Create params for the provider
	_, seg, err = capnp.NewMessage(capnp.SingleSegment(nil))
	if err != nil {
		return errnie.Error(err)
	}

	params, err := params.NewRootParams(seg)
	if err != nil {
		return errnie.Error(err)
	}

	// Process through provider
	for artifact := range prvdr.Generate(params, context, mcpTools) {
		payload, err := artifact.DecryptPayload()

		if err != nil {
			return errnie.Error(err)
		}

		// Send response back through RPC
		response := datura.New(
			datura.WithRole(datura.ArtifactRoleAnswer),
			datura.WithScope(datura.ArtifactScopeGeneration),
			datura.WithPayload(payload),
		)

		// Send response back through RPC
		if err := srv.processMessage(ctx, "response", response); err != nil {
			return errnie.Error(err)
		}
	}

	return nil
}

// handleData processes data/information from other agents
func (srv *AgentServer) handleData(
	ctx context.Context,
	msg *datura.ArtifactBuilder,
) error {
	// TODO: Implement data handling based on agent role
	return nil
}

// processMessage handles topic-based message routing
func (srv *AgentServer) processMessage(
	ctx context.Context,
	topic string,
	msg *datura.ArtifactBuilder,
) error {
	// TODO: Implement topic-based message processing
	return nil
}
