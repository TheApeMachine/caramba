package agent

import (
	"context"
	"fmt"
	"io"

	"capnproto.org/go/capnp/v3"
	"capnproto.org/go/capnp/v3/rpc"
	"github.com/davecgh/go-spew/spew"
	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

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

func (agent *Agent) Client() AgentRPC {
	return AgentToClient(agent)
}

func (agent *Agent) Conn(transport io.ReadWriteCloser) *rpc.Conn {
	return rpc.NewConn(rpc.NewStreamTransport(transport), &rpc.Options{
		BootstrapClient: capnp.Client(agent.Client()),
	})
}

// SendTask sends a task to the agent
func (agent *AgentBuilder) SendTask(task *datura.ArtifactBuilder) error {
	out := datura.New()

	// Get agent client from bootstrapped connection
	client := agent.Conn(out).Bootstrap(context.Background())
	agentClient := AgentRPC(client)

	// Marshal the task for transmission
	msgData, err := task.Message().Marshal()
	if err != nil {
		return errnie.Error(err)
	}

	// Send message using the process method
	future, release := agentClient.Process(context.Background(), func(p AgentRPC_process_Params) error {
		return p.SetMessage_(msgData)
	})
	defer release()

	// Wait for response
	result, err := future.Struct()
	if err != nil {
		return errnie.Error(err)
	}

	// Process response if needed
	response, err := result.Response()
	if err != nil {
		return errnie.Error(err)
	}

	spew.Dump(response)

	// Handle response if present
	if len(response) > 0 {
		if responseArtifact := datura.Unmarshal(response); responseArtifact != nil {
			if payload, err := responseArtifact.DecryptPayload(); err == nil {
				identity, err := agent.Identity()
				if err != nil {
					return errnie.Error(err)
				}
				name, err := identity.Name()
				if err != nil {
					return errnie.Error(err)
				}
				fmt.Printf("Response from %s: %s\n", name, string(payload))
			}
		}
	}

	return nil
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
