package agent

import (
	"io"

	"capnproto.org/go/capnp/v3"
	"capnproto.org/go/capnp/v3/rpc"
	"github.com/google/uuid"
	aictx "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
AgentBuilder wraps an Agent and provides various convenience methods
for building and managing agents.
*/
type AgentBuilder struct {
	*Agent
	Transport io.ReadWriteCloser
}

// AgentBuilderOption defines a function that configures an Agent
type AgentBuilderOption func(*AgentBuilder) error

// New creates a new agent with the provided options
func New(options ...AgentBuilderOption) *AgentBuilder {
	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		agent Agent
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if agent, err = NewRootAgent(seg); errnie.Error(err) != nil {
		return nil
	}

	// Create and set identity with default ID
	identity, err := agent.NewIdentity()

	if errnie.Error(err) != nil {
		return nil
	}

	if errnie.Error(identity.SetIdentifier(uuid.New().String())) != nil {
		return nil
	}

	params, err := agent.NewParams()

	if errnie.Error(err) != nil {
		return nil
	}

	if errnie.Error(agent.SetParams(params)) != nil {
		return nil
	}

	agent.SetContext(*aictx.New().Context)

	builder := &AgentBuilder{
		Agent: &agent,
	}

	// Apply all options
	for _, option := range options {
		if err := option(builder); errnie.Error(err) != nil {
			return nil
		}
	}

	return builder
}

func (agent *Agent) Client() RPC {
	return AgentToClient(agent)
}

func (agent *Agent) Conn(transport io.ReadWriteCloser) *rpc.Conn {
	return rpc.NewConn(rpc.NewStreamTransport(transport), &rpc.Options{
		BootstrapClient: capnp.Client(agent.Client()),
	})
}

// WithName sets the agent's name
func WithName(name string) AgentBuilderOption {
	return func(a *AgentBuilder) error {
		identity, err := a.Identity()

		if err != nil {
			return errnie.Error(err)
		}

		return errnie.Error(identity.SetName(name))
	}
}

// WithRole sets the agent's role
func WithRole(role string) AgentBuilderOption {
	return func(a *AgentBuilder) error {
		identity, err := a.Identity()

		if err != nil {
			return errnie.Error(err)
		}

		return errnie.Error(identity.SetRole(role))
	}
}

func WithTransport(transport io.ReadWriteCloser) AgentBuilderOption {
	return func(a *AgentBuilder) error {
		a.Transport = transport
		return nil
	}
}

func WithModel(model string) AgentBuilderOption {
	return func(a *AgentBuilder) error {
		params, err := a.Params()
		if err != nil {
			return errnie.Error(err)
		}

		return errnie.Error(params.SetModel(model))
	}
}

// ProcessCommand handles command messages sent to the agent
func (agent *AgentBuilder) ProcessCommand(msg *datura.ArtifactBuilder) error {
	// Get command from message metadata
	cmd := datura.GetMetaValue[string](msg, "command")
	if cmd == "" {
		return nil // Skip messages without a command
	}

	// Process command based on type
	switch cmd {
	case "stop":
		// Handle stop command
		return nil
	default:
		// Unknown command
		return nil
	}
}

// UpdateStatus handles status update messages
func (agent *AgentBuilder) UpdateStatus(msg *datura.ArtifactBuilder) error {
	// Update agent status based on message
	status := datura.GetMetaValue[string](msg, "status")
	if status == "" {
		return nil // Skip messages without status
	}

	// Update agent state based on status
	return nil
}

// ProcessMessage handles general messages sent to the agent
func (agent *AgentBuilder) ProcessMessage(msg *datura.ArtifactBuilder) error {
	// Process message based on its role
	role := datura.GetMetaValue[string](msg, "role")
	if role == "" {
		return nil // Skip messages without role
	}

	// Get message payload
	_, err := msg.DecryptPayload()
	if err != nil {
		return err
	}

	// Add message to agent's context
	return nil
}

// IsActive checks if the agent is still active
func (agent *AgentBuilder) IsActive() bool {
	// Check agent's active status
	return true // For now, always return true
}

// Maintain performs periodic maintenance tasks
func (agent *AgentBuilder) Maintain() error {
	// Perform any necessary maintenance
	return nil
}
