package agent

import (
	context "context"
	"io"

	"capnproto.org/go/capnp/v3"
	"capnproto.org/go/capnp/v3/rpc"
	"github.com/google/uuid"
	aictx "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/message"
	"github.com/theapemachine/caramba/pkg/ai/prompt"
	prvdr "github.com/theapemachine/caramba/pkg/ai/provider"
	datura "github.com/theapemachine/caramba/pkg/datura"
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

func (builder *AgentBuilder) Send(message *message.MessageBuilder) *datura.ArtifactBuilder {
	future, release := builder.Client().Send(
		context.Background(), func(p RPC_send_Params) error {
			return nil
		},
	)

	defer release()

	var (
		result RPC_send_Results
		err    error
		out    datura.Artifact
	)

	if result, err = future.Struct(); errnie.Error(err) != nil {
		return nil
	}

	out, err = result.Out()

	if errnie.Error(err) != nil {
		return nil
	}

	return datura.New(datura.WithArtifact(&out))
}

func (builder *AgentBuilder) Identity() (string, string) {
	identity, err := builder.Agent.Identity()

	if errnie.Error(err) != nil {
		return "noid", "noname"
	}

	id, err := identity.Identifier()

	if errnie.Error(err) != nil {
		return "noid", "noname"
	}

	name, err := identity.Name()

	if errnie.Error(err) != nil {
		return "noid", "noname"
	}

	return id, name
}

func (builder *AgentBuilder) Role() string {
	identity, err := builder.Agent.Identity()

	if errnie.Error(err) != nil {
		return ""
	}

	role, err := identity.Role()

	if errnie.Error(err) != nil {
		return ""
	}

	return role
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
		identity, err := a.Agent.Identity()

		if err != nil {
			return errnie.Error(err)
		}

		return errnie.Error(identity.SetName(name))
	}
}

// WithRole sets the agent's role
func WithRole(role string) AgentBuilderOption {
	return func(a *AgentBuilder) error {
		identity, err := a.Agent.Identity()

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

func WithProvider(provider string) AgentBuilderOption {
	return func(a *AgentBuilder) error {
		p := prvdr.New(prvdr.WithName(provider))

		if err := a.SetProvider(*p.Provider); err != nil {
			return errnie.Error(err)
		}

		return nil
	}
}

func WithPrompt(role string, prompt *prompt.PromptBuilder) AgentBuilderOption {
	return func(a *AgentBuilder) error {
		ctx, err := a.Context()

		if errnie.Error(err) != nil {
			return errnie.Error(err)
		}

		messages := errnie.Try(ctx.Messages())
		ml := errnie.Try(message.NewMessage_List(a.Segment(), int32(messages.Len()+1)))

		for i := range messages.Len() {
			ml.Set(i, messages.At(i))
		}

		msg := message.New(
			message.WithRole(role),
			message.WithContent(prompt.String()),
		)

		if errnie.Error(ml.Set(messages.Len(), *msg.Message)) != nil {
			return errnie.Error(err)
		}

		if errnie.Error(ctx.SetMessages(ml)) != nil {
			return errnie.Error(err)
		}

		return nil
	}
}
