package ai

import (
	"context"
	"errors"
	"slices"
	"time"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/system"
	"github.com/theapemachine/caramba/pkg/utils"
)

/*
Builder wraps a Cap'n Proto Agent.
*/
type AgentBuilder struct {
	*Agent
	pctx          context.Context
	ctx           context.Context
	cancel        context.CancelFunc
	status        core.Status
	waiters       []datura.ArtifactScope
	out           chan *datura.Artifact
	paramsBuilder *core.ParamsBuilder
	ctxBuilder    *core.ContextBuilder
	tools         []string
	protocol      *core.Protocol
}

type AgentOption func(*AgentBuilder)

/*
NewAgentBuilder creates a new agent with initialized components.
*/
func NewAgentBuilder(options ...AgentOption) *AgentBuilder {
	errnie.Debug("ai.NewAgentBuilder")

	var (
		cpnp  = utils.NewCapnp()
		agent Agent
		err   error
	)

	if agent, err = NewRootAgent(cpnp.Seg); errnie.Error(err) != nil {
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	builder := &AgentBuilder{
		Agent:  &agent,
		ctx:    ctx,
		cancel: cancel,
	}

	for _, option := range options {
		option(builder)
	}

	return builder
}

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

func (builder *AgentBuilder) Generate(
	buffer chan *datura.Artifact,
	fn ...func(*datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("ai.AgentBuilder.Generate")

	if err := builder.Validate("ai.AgentBuilder.Generate"); err != nil {
		return nil
	}

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

				if err := builder.handleArtifact(artifact); err != nil {
					errnie.Error(err)
					continue
				}
			case <-time.After(100 * time.Millisecond):
				// Do nothing
			}
		}
	}()

	return builder.out
}

// handleArtifact processes a single artifact according to the protocol
func (builder *AgentBuilder) handleArtifact(artifact *datura.Artifact) error {
	if builder.protocol == nil {
		builder.protocol = system.NewHub().RegisterProtocol(
			core.NewProtocol(
				datura.GetMetaValue[string](artifact, "topic"),
				builder.ID(),
				"provider",
			),
		)
	}

	if len(builder.waiters) > 0 && !slices.Contains(builder.waiters, datura.ArtifactScope(artifact.Scope())) {
		return nil
	}

	builder.waiters = slices.DeleteFunc(builder.waiters, func(scope datura.ArtifactScope) bool {
		return scope == datura.ArtifactScope(artifact.Scope())
	})

	var step *datura.Artifact
	step, builder.status = builder.protocol.HandleMessage(builder.ID(), artifact)

	switch builder.status {
	case core.StatusWorking:
		builder.handlePreflight()
	case core.StatusWaiting:
		builder.waiters = append(builder.waiters, datura.ArtifactScope(step.Scope()))
	case core.StatusDone:
		// Send acknowledgment and release
		ack := datura.New(
			datura.WithRole(datura.ArtifactRoleAcknowledge),
			datura.WithScope(datura.ArtifactScopeRelease),
			datura.WithMeta("from", builder.ID()),
			datura.WithMeta("to", "provider"),
			datura.WithMeta("protocol", datura.GetMetaValue[string](step, "protocol")),
		)

		builder.out <- ack
	default:
		errnie.Debug("ai.AgentBuilder.handleArtifact", "status", builder.status)
		builder.out <- step
	}

	return nil
}

func (builder *AgentBuilder) handlePreflight() {
	builder.out <- builder.paramsBuilder.Artifact()
	builder.out <- builder.ctxBuilder.Artifact()
}

func (builder *AgentBuilder) Validate(scope string) error {
	errnie.Debug("ai.AgentBuilder.Validate")

	if builder.Agent == nil {
		return NewAgentValidationError(scope, errors.New("agent not set"))
	}

	if builder.pctx == nil {
		return NewAgentValidationError(scope, errors.New("parent context not set"))
	}

	if builder.ctx == nil {
		return NewAgentValidationError(scope, errors.New("context not set"))
	}

	if builder.cancel == nil {
		return NewAgentValidationError(scope, errors.New("cancel function not set"))
	}

	return nil
}

func WithCancel(ctx context.Context) AgentOption {
	return func(builder *AgentBuilder) {
		builder.pctx = ctx
	}
}

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

func WithParams(params *core.ParamsBuilder) AgentOption {
	return func(builder *AgentBuilder) {
		builder.paramsBuilder = params
	}
}

func WithContext(context *core.ContextBuilder) AgentOption {
	return func(builder *AgentBuilder) {
		builder.ctxBuilder = context
	}
}

func WithTools(tools ...string) AgentOption {
	return func(builder *AgentBuilder) {
		builder.tools = tools
	}
}
