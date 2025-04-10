package service

import (
	"context"
	"sync"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
)

type ProviderComponent struct {
	ctx           context.Context
	cancel        context.CancelFunc
	wg            *sync.WaitGroup
	id            string
	name          string
	subscriptions []string
}

func NewProviderComponent(
	pctx context.Context,
	wg *sync.WaitGroup,
	name string,
	subscriptions []string,
) (*ProviderComponent, error) {
	errnie.Trace("NewProviderComponent")

	ctx, cancel := context.WithCancel(pctx)

	component := &ProviderComponent{
		ctx:           ctx,
		cancel:        cancel,
		wg:            wg,
		id:            uuid.New().String(),
		name:          name,
		subscriptions: subscriptions,
	}

	service, err := NewComponentService(
		ctx, wg, name, component, subscriptions,
	)

	if err != nil {
		return nil, errnie.InternalError(err)
	}

	service.Start()

	return component, nil
}

func (component *ProviderComponent) ID() string {
	return component.id
}

func (component *ProviderComponent) Name() string {
	return component.name
}

func (component *ProviderComponent) HandleMessage(
	artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("ProviderComponent.HandleMessage")

	switch datura.ArtifactRole(artifact.Role()) {
	case datura.ArtifactRolePublisher:
		component.handlePublisher(artifact)
	}

	return artifact
}

func (component *ProviderComponent) handlePublisher(
	artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("ProviderComponent.handlePublisher")

	switch datura.ArtifactScope(artifact.Scope()) {
	case datura.ArtifactScopeGeneration:
		switch datura.GetMetaValue[string](artifact, "to") {
		case "provider.openai":
			return provider.NewOpenAIProvider().Generate(
				component.ctx, artifact,
			)
		}
	}

	return datura.New(
		datura.WithScope(datura.ArtifactScopeUnknown),
		datura.WithRole(datura.ArtifactRoleUnknown),
		datura.WithBytes([]byte("")),
	)
}
