package provider

import (
	"context"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/utils"
)

type ProviderInterface interface {
	stream.Generator
	Name() string
}

type ProviderType ProviderInterface

var (
	ProviderTypeMock      ProviderType = NewMockProvider()
	ProviderTypeOpenAI    ProviderType = NewOpenAIProvider()
	ProviderTypeAnthropic ProviderType = NewAnthropicProvider()
	ProviderTypeGoogle    ProviderType = NewGoogleProvider()
	ProviderTypeCohere    ProviderType = NewCohereProvider()
	ProviderTypeDeepSeek  ProviderType = NewDeepseekProvider()
	ProviderTypeOllama    ProviderType = NewOllamaProvider()
)

type ProviderBuilder struct {
	*Provider
	pctx     context.Context
	ctx      context.Context
	cancel   context.CancelFunc
	supplier ProviderType
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
	return builder.supplier.Generate(buffer, fn...)
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
