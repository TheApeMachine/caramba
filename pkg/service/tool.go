package service

import (
	"context"
	"sync"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type ToolComponent struct {
	ctx           context.Context
	cancel        context.CancelFunc
	wg            *sync.WaitGroup
	id            string
	name          string
	subscriptions []string
}

func NewToolComponent(
	pctx context.Context,
	wg *sync.WaitGroup,
	name string,
	subscriptions []string,
) (*ToolComponent, error) {
	errnie.Trace("NewToolComponent")

	ctx, cancel := context.WithCancel(pctx)

	component := &ToolComponent{
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

func (component *ToolComponent) ID() string {
	return component.id
}

func (component *ToolComponent) Name() string {
	return component.name
}

func (component *ToolComponent) HandleMessage(artifact *datura.Artifact) *datura.Artifact {
	errnie.Trace("ToolComponent.HandleMessage")

	switch datura.ArtifactRole(artifact.Role()) {
	case datura.ArtifactRolePublisher:
		component.handlePublisher(artifact)
	}

	return artifact
}

func (component *ToolComponent) handlePublisher(
	artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("ToolComponent.handlePublisher")

	switch datura.ArtifactScope(artifact.Scope()) {
	case datura.ArtifactScopeError:
		// TODO: Handle error
	}

	return datura.New(
		datura.WithScope(datura.ArtifactScopeUnknown),
		datura.WithRole(datura.ArtifactRoleUnknown),
		datura.WithBytes([]byte("")),
	)
}
