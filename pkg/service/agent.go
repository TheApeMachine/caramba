package service

import (
	"context"
	"sync"

	"github.com/google/uuid"
	"github.com/theapemachine/caramba/pkg/ai/agent"
	"github.com/theapemachine/caramba/pkg/ai/prompt"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type AgentComponent struct {
	ctx           context.Context
	cancel        context.CancelFunc
	wg            *sync.WaitGroup
	id            string
	name          string
	tools         []string
	subscriptions []string
	agent         *agent.Agent
}

func NewAgentComponent(
	pctx context.Context,
	wg *sync.WaitGroup,
	name string,
	initialMessage string,
	tools []string,
	subscriptions []string,
) (*AgentComponent, error) {
	errnie.Trace("AgentComponent.NewAgentComponent")
	ctx, cancel := context.WithCancel(pctx)

	component := &AgentComponent{
		ctx:           ctx,
		cancel:        cancel,
		wg:            wg,
		id:            uuid.New().String(),
		name:          name,
		tools:         tools,
		subscriptions: subscriptions,
		agent: agent.New(
			agent.WithName(name),
			agent.WithTools(tools...),
			agent.WithPrompt("system", prompt.New(
				prompt.WithFragments(
					prompt.NewFragmentBuilder(
						prompt.WithBuiltin(name),
					),
				),
			)),
		),
	}

	service, err := NewComponentService(
		ctx, wg, name, component, subscriptions,
	)

	if err != nil {
		return nil, errnie.InternalError(err)
	}

	service.Start()

	if initialMessage != "" {
		service.transport.Publish(datura.New(
			datura.WithRole(datura.ArtifactRoleUser),
			datura.WithScope(datura.ArtifactScopeGeneration),
			datura.WithPayload([]byte(initialMessage)),
			datura.WithMeta("topic", "task"),
		))
	}

	return component, nil
}

func (component *AgentComponent) ID() string {
	return component.id
}

func (component *AgentComponent) Name() string {
	return component.name
}

func (component *AgentComponent) HandleMessage(
	artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("AgentComponent.HandleMessage")

	switch datura.ArtifactRole(artifact.Role()) {
	case datura.ArtifactRolePublisher:
		component.handlePublisher(artifact)
	case datura.ArtifactRoleUser:
		component.handleUser(artifact)
	}

	return artifact
}

func (component *AgentComponent) handlePublisher(
	artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("AgentComponent.handlePublisher")

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

func (component *AgentComponent) handleUser(
	artifact *datura.Artifact,
) *datura.Artifact {
	errnie.Trace("AgentComponent.handleUser")

	switch datura.ArtifactScope(artifact.Scope()) {
	case datura.ArtifactScopeGeneration:
		return datura.New(
			datura.WithRole(datura.ArtifactRolePublisher),
			datura.WithScope(datura.ArtifactScopeGeneration),
			datura.WithBytes(component.agent.Bytes()),
			datura.WithMeta("to", "provider.openai"),
		)
	}

	return artifact
}
