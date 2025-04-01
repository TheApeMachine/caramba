package ai

import (
	"context"
	"time"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/utils"
)

func init() {
	provider.RegisterTool("agent")
}

/*
Builder wraps a Cap'n Proto Agent.
*/
type AgentBuilder struct {
	*Agent
	pctx   context.Context
	ctx    context.Context
	cancel context.CancelFunc
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
	}

	for _, option := range options {
		option(builder)
	}

	return builder
}

func (builder *AgentBuilder) Generate(
	buffer chan *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("ai.AgentBuilder.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		for {
			select {
			case <-builder.pctx.Done():
				errnie.Info("AgentBuilder context cancelled")
				builder.cancel()
				return
			case <-builder.ctx.Done():
				errnie.Info("AgentBuilder context cancelled")
				builder.cancel()
				return
			case artifact := <-buffer:
				out <- artifact
			default:
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()

	return out
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

func WithProvider(prvdr provider.ProviderType) AgentOption {
	return func(builder *AgentBuilder) {
		pb := provider.NewProviderBuilder(
			provider.WithSupplier(prvdr.Name()),
		)

		if err := builder.SetProvider(*pb.Provider); errnie.Error(err) != nil {
			return
		}
	}
}

func WithParams(params *ParamsBuilder) AgentOption {
	return func(builder *AgentBuilder) {
		if err := builder.SetParams(*params.Params); errnie.Error(err) != nil {
			return
		}
	}
}

func WithContext(context *ContextBuilder) AgentOption {
	return func(builder *AgentBuilder) {
		if err := builder.SetContext(*context.Context); errnie.Error(err) != nil {
			return
		}
	}
}

func WithTools(tools ...*tools.ToolBuilder) AgentOption {
	return func(builder *AgentBuilder) {
		toolList, err := builder.NewTools(int32(len(tools)))

		if errnie.Error(err) != nil {
			return
		}

		for i, t := range tools {
			fn, err := t.Function()

			if errnie.Error(err) != nil {
				return
			}

			tl := toolList.At(i)
			tl.SetFunction(fn)
		}

		if err := builder.SetTools(toolList); errnie.Error(err) != nil {
			return
		}
	}
}
