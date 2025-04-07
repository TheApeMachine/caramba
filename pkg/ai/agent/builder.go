package agent

import (
	context "context"
	"fmt"
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
	tool "github.com/theapemachine/caramba/pkg/ai/tool"
	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	aitool "github.com/theapemachine/caramba/pkg/tools"
)

/*
AgentBuilder wraps an Agent and provides various convenience methods
for building and managing agents.
*/
type AgentBuilder struct {
	*Agent     `json:"agent"`
	AIParams   *params.ParamsBuilder  `json:"ai_params"`
	AIContext  *aictx.ContextBuilder  `json:"ai_context"`
	AIProvider *prvdr.ProviderBuilder `json:"ai_provider"`
	Transport  io.ReadWriteCloser     `json:"transport"`
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

	// Create artifact with context payload
	artifact := datura.New(
		datura.WithPayload(builder.AIContext.Bytes()),
		datura.WithMeta("model", errnie.Try(errnie.Try(builder.Params()).Model())),
		datura.WithMeta("temperature", errnie.Try(builder.Params()).Temperature()),
		datura.WithMeta("top_p", errnie.Try(builder.Params()).TopP()),
	)

	fmt.Println(string(artifact.Bytes()))

	// Add tools to artifact metadata if they exist
	if tools, err := builder.Tools(); err == nil && tools.Len() > 0 {
		var toolTypes []aitool.ToolType
		for i := 0; i < tools.Len(); i++ {
			// Since we're working with Tool_List, we need to convert each Tool to ToolBuilder
			toolBuilder := tool.New(tool.WithBytes(tools.At(i).ToPtr().EncodeAsPtr(tools.At(i).Segment()).Data()))
			if toolBuilder != nil && toolBuilder.MCPTool != nil {
				toolTypes = append(toolTypes, *toolBuilder.MCPTool)
			}
		}
		if len(toolTypes) > 0 {
			artifact.SetMetaValue("tools", toolTypes)
		}
	}

	fmt.Println(*artifact)

	// Generate response using the provider
	response := builder.AIProvider.Generate(context.Background(), artifact)
	return *response.Artifact
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

func WithProvider(provider *prvdr.ProviderBuilder) AgentBuilderOption {
	return func(a *AgentBuilder) error {
		a.AIProvider = provider

		if err := a.SetProvider(*a.AIProvider.Provider); err != nil {
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

func WithTools(tools ...*tool.ToolBuilder) AgentBuilderOption {
	return func(a *AgentBuilder) error {
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
