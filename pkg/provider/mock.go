package provider

import (
	"context"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/utils"
)

type MockProvider struct {
	*Provider
	params *Params
	ctx    context.Context
	cancel context.CancelFunc
}

func NewMockProvider(opts ...MockProviderOption) *MockProvider {
	errnie.Debug("provider.NewMockProvider")

	var (
		cpnp     = utils.NewCapnp()
		provider Provider
		err      error
	)

	if provider, err = NewRootProvider(cpnp.Seg); errnie.Error(err) != nil {
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())

	prvdr := &MockProvider{
		Provider: &provider,
		params:   &Params{},
		ctx:      ctx,
		cancel:   cancel,
	}

	for _, opt := range opts {
		opt(prvdr)
	}

	return prvdr
}

func (prvdr *MockProvider) ID() string {
	return "mock"
}

type MockProviderOption func(*MockProvider)

func (prvdr *MockProvider) Generate(
	buffer chan *datura.Artifact,
	fn ...func(artifact *datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	errnie.Debug("provider.MockProvider.Generate")

	out := make(chan *datura.Artifact)

	go func() {
		defer close(out)

		select {
		case <-prvdr.ctx.Done():
			errnie.Debug("provider.MockProvider.Generate.ctx.Done")
			prvdr.cancel()
			return
		case artifact := <-buffer:
			if err := artifact.To(prvdr.params); err != nil {
				out <- datura.New(datura.WithError(errnie.Error(err)))
				return
			}

			// For mock provider, just echo back the received message
			out <- datura.New(datura.WithPayload(prvdr.params.Marshal()))
		}
	}()

	return out
}

func (prvdr *MockProvider) Name() string {
	return "mock"
}
