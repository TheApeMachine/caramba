package provider

import (
	"context"
	"slices"
	"time"

	"github.com/theapemachine/caramba/pkg/core"
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
	status   core.Status
	waiters  []datura.ArtifactScope
	out      chan *datura.Artifact
	supplier string
	params   *datura.Artifact
	context  *datura.Artifact
}

type ProviderOption func(*ProviderBuilder)

func NewProviderBuilder(options ...ProviderOption) *ProviderBuilder {
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

func (builder *ProviderBuilder) ID() string {
	return utils.GenerateName()
}

func (builder *ProviderBuilder) Generate(
	buffer chan *datura.Artifact,
	fn ...func(*datura.Artifact) *datura.Artifact,
) chan *datura.Artifact {
	builder.out = make(chan *datura.Artifact)

	go func() {
		defer close(builder.out)

		for {
			select {
			case <-builder.pctx.Done():
				errnie.Info("ProviderBuilder context cancelled")
				builder.cancel()
				return
			case <-builder.ctx.Done():
				errnie.Info("ProviderBuilder context cancelled")
				return
			case artifact, ok := <-buffer:
				if !ok {
					return
				}

				if builder.status != core.StatusReady && builder.status != core.StatusWaiting {
					break
				}

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

func (builder *ProviderBuilder) handleWaiting(artifact *datura.Artifact) {
	if slices.Contains(builder.waiters, datura.ArtifactScope(artifact.Scope())) {
		index := slices.Index(builder.waiters, datura.ArtifactScope(artifact.Scope()))
		if index >= 0 {
			builder.waiters = slices.Delete(builder.waiters, index, index+1)

			switch datura.ArtifactScope(artifact.Scope()) {
			case datura.ArtifactScopeParams:
				builder.params = artifact
			case datura.ArtifactScopeContext:
				builder.context = artifact
			}
		}
	}

	if len(builder.waiters) == 0 {
		builder.status = core.StatusReady
		builder.waiters = nil

		// Process the received params and context to generate a response
		switch builder.supplier {
		case "openai":
			prvdr := NewOpenAIProvider()
			ch := make(chan *datura.Artifact)
			go func() {
				defer close(ch)
				ch <- builder.params
				ch <- builder.context
			}()
			for artifact := range prvdr.Generate(ch) {
				builder.out <- artifact
			}
		}
	}
}

func (builder *ProviderBuilder) handleAnswer(artifact *datura.Artifact) {
	switch datura.ArtifactScope(artifact.Scope()) {
	case datura.ArtifactScopeParams:
		builder.params = core.NewParamsBuilder().WithArtifact(artifact).Artifact()
	case datura.ArtifactScopeContext:
		builder.context = core.NewContextBuilder().WithArtifact(artifact).Artifact()
	}
}

func (builder *ProviderBuilder) handleAcknowledge(artifact *datura.Artifact) {
	builder.out <- artifact
}

func (builder *ProviderBuilder) handleQuestion(artifact *datura.Artifact) {
	switch datura.ArtifactScope(artifact.Scope()) {
	case datura.ArtifactScopeAquire:
		builder.out <- datura.New(
			datura.WithRole(datura.ArtifactRoleAcknowledge),
			datura.WithScope(datura.ArtifactScopeAquire),
		)

		builder.status = core.StatusWaiting
		builder.waiters = append(builder.waiters, datura.ArtifactScopeParams)
		builder.waiters = append(builder.waiters, datura.ArtifactScopeContext)
	}
}

func WithCancel(ctx context.Context) ProviderOption {
	return func(builder *ProviderBuilder) {
		builder.pctx = ctx
	}
}

func WithSupplier(name string) ProviderOption {
	return func(builder *ProviderBuilder) {
		builder.supplier = name
	}
}
