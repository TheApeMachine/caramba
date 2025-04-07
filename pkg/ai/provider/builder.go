package provider

import (
	context "context"

	"capnproto.org/go/capnp/v3"
	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	aiprvdr "github.com/theapemachine/caramba/pkg/provider"
)

/*
ProviderBuilder wraps around a Provider and decorates it with
additional functionality, making it more convenient to use.
*/
type ProviderBuilder struct {
	*Provider
	AIProvider aiprvdr.ProviderType
	client     RPC
}

type ProviderBuilderOption func(*ProviderBuilder)

/*
New creates a new Provider, wrapped in a ProviderBuilder.
*/
func New(opts ...ProviderBuilderOption) *ProviderBuilder {
	errnie.Trace("provider.New")

	var (
		arena = capnp.SingleSegment(nil)
		seg   *capnp.Segment
		prvdr Provider
		err   error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return nil
	}

	if prvdr, err = NewRootProvider(seg); errnie.Error(err) != nil {
		return nil
	}

	builder := &ProviderBuilder{
		Provider: &prvdr,
	}

	for _, opt := range opts {
		opt(builder)
	}

	builder.client = ProviderToClient(builder)

	return builder
}

func (prvdr *ProviderBuilder) Generate(ctx context.Context, artifact *datura.ArtifactBuilder) *datura.ArtifactBuilder {
	errnie.Trace("provider.Generate")

	future, release := prvdr.client.Generate(
		ctx, func(p RPC_generate_Params) error {
			return p.SetArtifact(*artifact.Artifact)
		},
	)

	defer release()

	var (
		result RPC_generate_Results
		err    error
	)

	if result, err = future.Struct(); errnie.Error(err) != nil {
		return nil
	}

	out, err := result.Out()

	if errnie.Error(err) != nil {
		return nil
	}

	return datura.New(datura.WithArtifact(&out))
}

/*
WithName sets the name of the provider.
The name serves as the key for the LLM provider.
*/
func WithName(name string) ProviderBuilderOption {
	errnie.Trace("provider.WithName")

	return func(p *ProviderBuilder) {
		if err := p.SetName(name); err != nil {
			errnie.Error(err)
		}
	}
}

func WithAIProvider(name string, provider aiprvdr.ProviderType) ProviderBuilderOption {
	errnie.Trace("provider.WithAIProvider")

	return func(p *ProviderBuilder) {
		p.AIProvider = provider
		if err := p.SetName(name); errnie.Error(err) != nil {
			errnie.Error(err)
		}
	}
}
