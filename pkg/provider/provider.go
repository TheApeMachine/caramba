package provider

import (
	"context"
	"time"

	"github.com/theapemachine/caramba/pkg/core"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/protocol"
	"github.com/theapemachine/caramba/pkg/system"
	"github.com/theapemachine/caramba/pkg/utils"
)

type ProviderInterface interface {
	Name() string
	Handle(chan *datura.Artifact, *datura.Artifact) *datura.Artifact
}

type ProviderType ProviderInterface

var (
	ProviderTypeOpenAI ProviderType = NewOpenAIProvider()
)

type ProviderBuilder struct {
	*Provider
	pctx     context.Context
	ctx      context.Context
	cancel   context.CancelFunc
	supplier ProviderType
	out      chan *datura.Artifact
	status   core.Status
	protocol *protocol.Spec
}

type ProviderOption func(*ProviderBuilder)

func NewProviderBuilder(options ...ProviderOption) *ProviderBuilder {
	errnie.Debug("provider.NewProviderBuilder")

	var (
		cpnp     = utils.NewCapnp()
		provider Provider
		err      error
	)

	if provider, err = NewRootProvider(cpnp.Seg); errnie.Error(err) != nil {
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	builder := &ProviderBuilder{
		Provider: &provider,
		ctx:      ctx,
		cancel:   cancel,
	}

	for _, option := range options {
		option(builder)
	}

	return builder
}

func (builder *ProviderBuilder) Generate(
	buffer chan *datura.Artifact,
	fn ...func(*datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("provider.ProviderBuilder.Generate")

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
					builder.protocol = system.NewHub().GetProtocol(
						datura.GetMetaValue[string](artifact, "protocol"),
					)
				}

				out, status := builder.protocol.Next(artifact, builder.status)

				if builder.status == core.StatusWaiting && status == core.StatusWaiting {
					// We should not continue to the next step until the
					// waiting conditions have been resolved by the other party.
					continue
				}

				builder.supplier.Handle(builder.out, out)
				builder.status = status
				errnie.Info("provider.ProviderBuilder.Generate.Status", "status", builder.status.String())
			case <-time.After(100 * time.Millisecond):
				// Do nothing
			}
		}
	}()

	return builder.out

}

func WithCancel(ctx context.Context) ProviderOption {
	return func(builder *ProviderBuilder) {
		builder.pctx = ctx
	}
}

func WithSupplier(supplier ProviderType) ProviderOption {
	return func(builder *ProviderBuilder) {
		builder.supplier = supplier
	}
}

func (builder *ProviderBuilder) ID() string {
	return "provider"
}
