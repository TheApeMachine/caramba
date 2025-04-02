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
}

type AgentOption func(*AgentBuilder)

/*
NewAgentBuilder creates a new agent with initialized components.
*/
func NewAgentBuilder(options ...AgentOption) *AgentBuilder {
	errnie.Debug("NewBuilder")

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
		status: core.StatusUnknown,
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

	builder.out = make(chan *datura.Artifact)

	go func() {
		defer close(builder.out)

		for {
			select {
			case <-builder.pctx.Done():
				errnie.Info("AgentBuilder context cancelled")

				builder.status = core.StatusDone
				builder.cancel()

				return
			case <-builder.ctx.Done():
				errnie.Info("AgentBuilder context cancelled")
				builder.status = core.StatusDone
				return
			case artifact, ok := <-buffer:
				if !ok {
					builder.status = core.StatusError
					return
				}

				// Make sure we are either ready for work, or waiting for a specific artifact.
				if builder.status != core.StatusReady && builder.status != core.StatusWaiting {
					break
				}

				// If we are waiting for a specific artifact, we should not be doing anything else.
				if builder.status == core.StatusWaiting {
					builder.handleWaiting(artifact)
					break
				}

				switch datura.ArtifactRole(artifact.Role()) {
				case datura.ArtifactRoleAnswer:
					builder.handleAnswer(artifact)
				case datura.ArtifactRoleAcknowledge:
					builder.handleAcknowledge(artifact)
				case datura.ArtifactRoleQuestion:
					builder.handleQuestion(artifact)
				}
			default:
				time.Sleep(10 * time.Millisecond)
			}
		}
	}()

	return builder.out
}

func (builder *AgentBuilder) handleWaiting(artifact *datura.Artifact) {
	if slices.Contains(builder.waiters, datura.ArtifactScope(artifact.Scope())) {
		index := slices.Index(builder.waiters, datura.ArtifactScope(artifact.Scope()))
		if index >= 0 {
			builder.waiters = slices.Delete(builder.waiters, index, index+1)
		}
	}

	if len(builder.waiters) == 0 {
		builder.status = core.StatusReady
		builder.waiters = nil
	}
}

func (builder *AgentBuilder) handleAnswer(artifact *datura.Artifact) {
	switch datura.ArtifactScope(artifact.Scope()) {
	case datura.ArtifactScopeGeneration:
		builder.status = core.StatusWorking

		builder.ctxBuilder.AddMessage(artifact)

		builder.out <- datura.New(
			datura.WithRole(datura.ArtifactRoleQuestion),
			datura.WithScope(datura.ArtifactScopeAquire),
		)

		builder.status = core.StatusWaiting
		builder.waiters = append(builder.waiters, datura.ArtifactScopeAquire)
	}
}

func (builder *AgentBuilder) handleAcknowledge(artifact *datura.Artifact) {
	builder.ctxBuilder.AddMessage(artifact)
	builder.status = core.StatusReady
	builder.waiters = append(builder.waiters, datura.ArtifactScopeUnknown)
}

func (builder *AgentBuilder) handleQuestion(artifact *datura.Artifact) {
	builder.ctxBuilder.AddMessage(artifact)

	builder.status = core.StatusWorking

	switch datura.ArtifactScope(artifact.Scope()) {
	case datura.ArtifactScopePreflight:
		builder.out <- builder.paramsBuilder.Artifact()
		builder.out <- builder.ctxBuilder.Artifact()
	}
}

func (builder *AgentBuilder) Validate(scope string) error {
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
