package provider

import (
	context "context"

	"capnproto.org/go/capnp/v3"
	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
ProviderBuilder wraps around a Provider and decorates it with
additional functionality, making it more convenient to use.
*/
type ProviderBuilder struct {
	*Provider
}

type ProviderBuilderOption func(*ProviderBuilder)

/*
New creates a new Provider, wrapped in a ProviderBuilder.
*/
func New(opts ...ProviderBuilderOption) *ProviderBuilder {
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

	return builder
}

func (prvdr *Provider) Generate(
	ctx context.Context,
	artifact *datura.ArtifactBuilder,
) *datura.ArtifactBuilder {
	future, release := prvdr.Client().Generate(
		ctx, func(p RPC_generate_Params) error {
			return p.SetContext(*artifact.Artifact)
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

func (builder *ProviderBuilder) Generate(
	ctx context.Context,
	artifact *datura.ArtifactBuilder,
) *datura.ArtifactBuilder {
	future, release := builder.Client().Generate(
		ctx, func(p RPC_generate_Params) error {
			return p.SetContext(*artifact.Artifact)
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
	return func(p *ProviderBuilder) {
		if err := p.SetName(name); err != nil {
			errnie.Error(err)
		}
	}
}

// Client converts the provider to an RPC client
func (p *Provider) Client() RPC {
	return ProviderToClient(p)
}

// Client converts the provider builder to an RPC client
func (p *ProviderBuilder) Client() RPC {
	return ProviderToClient(p.Provider)
}
