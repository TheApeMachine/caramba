/*
Package ai provides artificial intelligence capabilities for the Caramba system,
including agent-based processing and protocol handling.
*/
package ai

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/protocol"
	"github.com/theapemachine/caramba/pkg/system"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
AgentBuilder is a builder pattern implementation for creating and configuring AI agents.
It wraps a Cap'n Proto Agent and provides methods for setting up the agent's context,
parameters, and processing capabilities.

The builder maintains internal state including:
  - Parent and local contexts with cancellation
  - Processing status
  - Output channel for artifacts
  - Parameter and context builders
  - Available tools
  - Protocol specification
*/
type AgentBuilder struct {
	*Agent
	pctx          context.Context
	ctx           context.Context
	cancel        context.CancelFunc
	status        core.Status
	out           chan *datura.Artifact
	paramsBuilder *core.ParamsBuilder
	ctxBuilder    *core.ContextBuilder
	tools         []string
	protocol      *protocol.Spec
}

// AgentOption defines a function type for configuring an AgentBuilder.
// This allows for flexible and extensible agent configuration using the functional options pattern.
type AgentOption func(*AgentBuilder)

/*
NewAgentBuilder creates a new agent with initialized components.
It accepts a variadic number of AgentOption functions that can be used to configure
the agent during creation.

Returns a pointer to the configured AgentBuilder or nil if initialization fails.
*/
func NewAgentBuilder(options ...AgentOption) *AgentBuilder {
	errnie.Debug("ai.NewAgentBuilder")

	agent, err := NewRootAgent(utils.NewCapnp().Seg)
	if errnie.Error(err) != nil {
		return nil
	}

	builder := &AgentBuilder{
		Agent:  &agent,
		ctx:    context.Background(),
		cancel: func() {},
	}

	for _, option := range options {
		option(builder)
	}

	return builder
}

/*
ID retrieves the unique identifier of the agent.
Returns an empty string if there's an error accessing the identity.
*/
func (builder *AgentBuilder) ID() string {
	identity, err := builder.Agent.Identity()
	if errnie.Error(err) != nil {
		return ""
	}

	id, err := identity.Identifier()
	if errnie.Error(err) != nil {
		return ""
	}

	return id
}

/*
Generate runs the agent as a concurrent artifact processing stream.
It sets up a processing pipeline that:
  - Listens for incoming artifacts on the provided buffer channel
  - Processes artifacts according to the configured protocol
  - Handles protocol state transitions
  - Outputs processed artifacts on the builder's output channel

Parameters:
  - buffer: Input channel for artifacts to process
  - fn: Optional slice of transformation functions to apply to artifacts

Returns a channel that emits processed artifacts.
*/
func (builder *AgentBuilder) Generate(
	buffer chan *datura.Artifact,
	fn ...func(*datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("ai.AgentBuilder.Generate")

	builder.out = make(chan *datura.Artifact, 64)

	go func() {
		defer close(builder.out)

		for {
			select {
			case <-builder.pctx.Done():
				errnie.Info("AgentBuilder context cancelled")
				builder.cancel()
				return
			case <-builder.ctx.Done():
				errnie.Info("AgentBuilder context cancelled")
				return
			case artifact, ok := <-buffer:
				if !ok {
					return
				}

				if builder.protocol == nil {
					builder.protocol = system.NewHub().RegisterProtocol(
						protocols[datura.GetMetaValue[string](artifact, "topic")](builder.ID()),
					)
				}

				var out *datura.Artifact
				out, builder.status = builder.protocol.Next(artifact)

				if builder.status == core.StatusWaiting {
					break
				}

				builder.out <- builder.handle(out)
			case <-time.After(100 * time.Millisecond):
				// Do nothing
			}
		}
	}()

	return builder.out
}

/*
handle is a method to add additional logic to the return value of a protocol step.
This is useful for when there is a request for data or in general something that
can only be done in the agent layer.

Parameters:
  - artifact: The artifact to be handled

Returns the processed artifact.
*/
func (builder *AgentBuilder) handle(artifact *datura.Artifact) *datura.Artifact {
	return artifact
}

/*
WithCancel is an AgentOption that configures the agent with a parent context.
This allows for external cancellation control of the agent's operations.

Parameters:
  - ctx: The parent context to use for cancellation
*/
func WithCancel(ctx context.Context) AgentOption {
	return func(builder *AgentBuilder) {
		builder.pctx = ctx
	}
}

/*
WithIdentity is an AgentOption that configures the agent's identity.
It sets up the agent with a unique identifier, name, and role.

Parameters:
  - name: The name to assign to the agent
  - role: The role to assign to the agent
*/
func WithIdentity(name string, role string) AgentOption {
	return func(builder *AgentBuilder) {
		identity, err := builder.NewIdentity()
		if errnie.Error(err) != nil {
			return
		}

		if err = identity.SetIdentifier(
			uuid.New().String(),
		); errnie.Error(err) != nil {
			return
		}

		if err = identity.SetName(name); errnie.Error(err) != nil {
			return
		}

		if err = identity.SetRole(role); errnie.Error(err) != nil {
			return
		}

		if err = builder.SetIdentity(identity); errnie.Error(err) != nil {
			return
		}
	}
}

/*
WithParams is an AgentOption that configures the agent with a parameter builder.
This allows for customization of the agent's operational parameters.

Parameters:
  - params: The parameter builder to use for configuration
*/
func WithParams(params *core.ParamsBuilder) AgentOption {
	return func(builder *AgentBuilder) {
		builder.paramsBuilder = params
	}
}

/*
WithContext is an AgentOption that configures the agent with a context builder.
This allows for customization of the agent's execution context.

Parameters:
  - context: The context builder to use for configuration
*/
func WithContext(context *core.ContextBuilder) AgentOption {
	return func(builder *AgentBuilder) {
		builder.ctxBuilder = context
	}
}

/*
WithTools is an AgentOption that configures the agent with a set of available tools.
This defines the capabilities available to the agent during operation.

Parameters:
  - tools: Variadic list of tool identifiers
*/
func WithTools(tools ...string) AgentOption {
	return func(builder *AgentBuilder) {
		builder.tools = tools
	}
}
