package agent

import (
	"bytes"
	context "context"
	"io"

	"errors"

	"capnproto.org/go/capnp/v3"
	"capnproto.org/go/capnp/v3/rpc"
	"github.com/google/uuid"
	aictx "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/message"
	params "github.com/theapemachine/caramba/pkg/ai/params"
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
	*Agent    `json:"agent"`
	AIParams  *params.ParamsBuilder `json:"ai_params"`
	AIContext *aictx.ContextBuilder `json:"ai_context"`
	Transport io.ReadWriteCloser    `json:"transport"`
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

	params := params.New()

	if errnie.Error(agent.SetParams(*params.Params)) != nil {
		return nil
	}

	// Initialize context
	agentCtx := aictx.New()
	if agentCtx == nil {
		errnie.Error(errors.New("failed to create context"))
		return nil
	}

	if err := agent.SetContext(*agentCtx.Context); errnie.Error(err) != nil {
		return nil
	}

	builder := &AgentBuilder{
		Agent:     &agent,
		AIParams:  params,
		AIContext: agentCtx,
	}

	// Apply all options
	for _, option := range options {
		if err := option(builder); errnie.Error(err) != nil {
			return nil
		}
	}

	return builder
}

func (builder *AgentBuilder) Send(message *message.MessageBuilder) datura.Artifact {
	errnie.Debug("agent.Send")

	// Add the message to the context first
	builder.AIContext.Add(message)

	// Update the agent's context
	if err := builder.SetContext(*builder.AIContext.Context); errnie.Error(err) != nil {
		return *datura.New(datura.WithError(errnie.Error(err))).Artifact
	}

	// Get the updated context
	buf := bytes.NewBuffer(nil)
	if _, err := io.Copy(buf, builder.AIContext); errnie.Error(err) != nil {
		return *datura.New(datura.WithError(errnie.Error(err))).Artifact
	}

	// Create artifact with context payload
	artifact := datura.New(
		datura.WithPayload(buf.Bytes()),
		datura.WithMeta("model", errnie.Try(errnie.Try(builder.Params()).Model())),
		datura.WithMeta("temperature", errnie.Try(builder.Params()).Temperature()),
		datura.WithMeta("top_p", errnie.Try(builder.Params()).TopP()),
	)

	provider := errnie.Try(builder.Provider())
	response, release := provider.Client().Generate(
		context.Background(), func(p prvdr.RPC_generate_Params) error {
			return p.SetContext(*artifact.Artifact)
		},
	)

	defer release()

	result := errnie.Try(response.Struct())
	outArtifactStruct := errnie.Try(result.Out())

	// The provider returns an artifact struct from the RPC response segment.
	// We need to copy its payload into a new artifact with a managed segment
	// to avoid segment lifetime issues when the caller accesses the payload later.
	newArtifactBuilder := datura.New(
		datura.WithPayload(errnie.Try(outArtifactStruct.Payload())),
		// Optionally copy metadata if needed, though payload seems the main issue
		// datura.WithMetadata(datura.GetMetadataMap(outArtifactStruct)),
	)

	// Add error handling for payload extraction and new artifact creation if necessary
	if newArtifactBuilder == nil {
		// Handle error, maybe return an artifact with an error payload
		err := errors.New("failed to create new artifact builder after RPC call")
		return *datura.New(datura.WithError(err)).Artifact
	}

	// Return the artifact struct value from the newly created builder.
	return *newArtifactBuilder.Artifact
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

func (agent *AgentBuilder) Client() RPC {
	return AgentToClient(agent)
}

func (agent *AgentBuilder) Conn(transport io.ReadWriteCloser) *rpc.Conn {
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
		msg := message.New(
			message.WithRole(role),
			message.WithContent(prompt.String()),
		)

		// Ensure the context is initialized
		if a.AIContext == nil {
			a.AIContext = aictx.New()
		}

		// Add the message and update the agent's context
		a.AIContext.Add(msg)
		return a.SetContext(*a.AIContext.Context)
	}
}
