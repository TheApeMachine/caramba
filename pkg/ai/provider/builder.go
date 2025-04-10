package provider

import (
	"capnproto.org/go/capnp/v3"

	"github.com/google/uuid"
	datura "github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ProviderOption func(Provider) Provider

/*
New creates a new Provider, wrapped in a ProviderBuilder.
*/
func New(opts ...ProviderOption) Provider {
	errnie.Trace("provider.New")

	var (
		arena   = capnp.SingleSegment(nil)
		seg     *capnp.Segment
		builder Provider
		err     error
	)

	if _, seg, err = capnp.NewMessage(arena); errnie.Error(err) != nil {
		return errnie.Try(NewProvider(seg)).ToState(errnie.StateError)
	}

	if builder, err = NewRootProvider(seg); errnie.Error(err) != nil {
		return errnie.Try(NewProvider(seg)).ToState(errnie.StateError)
	}

	builder.SetUuid(uuid.New().String())
	builder.SetState(uint64(errnie.StatePending))

	for _, opt := range opts {
		opt(builder)
	}

	return datura.Register(builder)
}

/*
WithName sets the name of the provider.
The name serves as the key for the LLM provider.
*/
func WithName(name string) ProviderOption {
	errnie.Trace("provider.WithName")

	return func(p Provider) Provider {
		if err := p.SetName(name); errnie.Error(err) != nil {
			return p
		}

		return p
	}
}

func WithAIProvider(name string) ProviderOption {
	errnie.Trace("provider.WithAIProvider")

	return func(p Provider) Provider {
		if err := p.SetName(name); errnie.Error(err) != nil {
			return p
		}

		return p
	}
}
