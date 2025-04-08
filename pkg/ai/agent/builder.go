package agent

import (
	"io"

	"capnproto.org/go/capnp/v3"
	"github.com/google/uuid"
	aictx "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/message"
	params "github.com/theapemachine/caramba/pkg/ai/params"
	"github.com/theapemachine/caramba/pkg/ai/prompt"
	prvdr "github.com/theapemachine/caramba/pkg/ai/provider"
	tool "github.com/theapemachine/caramba/pkg/ai/tool"
	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// AgentBuilderOption defines a function that configures an Agent
type AgentOption func(Agent) error

// New creates a new agent with the provided options
func New(options ...AgentOption) Agent {
	errnie.Trace("agent.New")

	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		agent Agent
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return errnie.Try(NewAgent(seg)).ToState(errnie.StateError)
	}

	if agent, err = NewRootAgent(seg); errnie.Error(err) != nil {
		return errnie.Try(NewAgent(seg)).ToState(errnie.StateError)
	}

	// Create and set identity with default ID
	identity, err := agent.NewIdentity()

	if errnie.Error(err) != nil {
		return errnie.Try(NewAgent(seg)).ToState(errnie.StateError)
	}

	if errnie.Error(identity.SetIdentifier(uuid.New().String())) != nil {
		return errnie.Try(NewAgent(seg)).ToState(errnie.StateError)
	}

	params := params.New()

	if errnie.Error(agent.SetParams(*params.Params)) != nil {
		return errnie.Try(NewAgent(seg)).ToState(errnie.StateError)
	}

	// Initialize context
	agentCtx := aictx.New()

	if err := agent.SetContext(agentCtx); errnie.Error(err) != nil {
		return errnie.Try(NewAgent(seg)).ToState(errnie.StateError)
	}

	for _, option := range options {
		if err := option(agent); errnie.Error(err) != nil {
			return errnie.Try(NewAgent(seg)).ToState(errnie.StateError)
		}
	}

	return datura.Register(agent)
}

func WithArtifact(artifact datura.Artifact) AgentOption {
	return func(a Agent) (err error) {
		errnie.Trace("agent.WithArtifact")
		if _, err = io.Copy(a, artifact); err != nil {
			return errnie.New(
				errnie.WithError(err),
				errnie.WithMessage("failed to copy artifact to agent"),
			)
		}

		return nil
	}
}

// WithName sets the agent's name
func WithName(name string) AgentOption {
	return func(a Agent) (err error) {
		errnie.Trace("agent.WithName")
		identity := errnie.Try(a.Identity())
		return errnie.Error(identity.SetName(name))
	}
}

// WithRole sets the agent's role
func WithRole(role string) AgentOption {
	return func(a Agent) (err error) {
		errnie.Trace("agent.WithRole")
		identity := errnie.Try(a.Identity())
		return errnie.Error(identity.SetRole(role))
	}
}

func WithTransport(transport io.ReadWriteCloser) AgentOption {
	return func(a Agent) (err error) {
		errnie.Trace("agent.WithTransport")

		return nil
	}
}

func WithModel(model string) AgentOption {
	return func(a Agent) (err error) {
		errnie.Trace("agent.WithModel")
		params := errnie.Try(a.Params())
		return errnie.Error(params.SetModel(model))
	}
}

func WithProvider(provider prvdr.Provider) AgentOption {
	return func(a Agent) (err error) {
		errnie.Trace("agent.WithProvider")
		return errnie.Error(a.SetProvider(provider))
	}
}

func WithPrompt(role string, prompt *prompt.PromptBuilder) AgentOption {
	return func(a Agent) (err error) {
		errnie.Trace("agent.WithPrompt")

		msg := datura.Register(message.New(
			message.WithRole(role),
			message.WithContent(prompt.String()),
		))

		// Ensure the context is initialized
		agentCtx := errnie.Try(a.Context())
		agentCtx.Add(msg)
		return errnie.Error(a.SetContext(agentCtx))
	}
}

func WithTools(tools ...*tool.ToolBuilder) AgentOption {
	return func(a Agent) (err error) {
		errnie.Trace("agent.WithTools")

		tl, err := tool.NewTool_List(a.Segment(), int32(len(tools)))

		if errnie.Error(err) != nil {
			return errnie.Error(err)
		}

		for i, t := range tools {
			tl.Set(i, *t.Tool)
		}

		if err := a.SetTools(tl); err != nil {
			return errnie.Error(err)
		}

		return nil
	}
}
