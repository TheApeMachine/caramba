package examples

import (
	"io"
	"os"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/system"
	"github.com/theapemachine/caramba/pkg/tools"
	"github.com/theapemachine/caramba/pkg/tweaker"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type MultiAgent struct {
	wf *workflow.Graph
}

func NewMultiAgent() *MultiAgent {
	errnie.Debug("NewMultiAgent")

	caller := tools.NewCaller()

	agent1 := ai.NewAgent(
		ai.WithCaller(caller),
	)

	msg := datura.New(
		datura.WithPayload(provider.NewParams(
			provider.WithModel("gpt-4o-mini"),
			provider.WithTopP(1),
			provider.WithTools(
				tools.NewSystemTool().Schema,
			),
			provider.WithMessages(
				provider.NewMessage(
					provider.WithSystemRole(
						tweaker.GetSystemPrompt("default"),
					),
				),
				provider.NewMessage(
					provider.WithUserRole(
						"Danny",
						"Try to find any agents you can communicate with, and send them a message.",
					),
				),
			),
		).Marshal()),
	)

	if _, err := io.Copy(agent1, msg); err != nil {
		errnie.Error(err)
		return nil
	}

	agent2 := ai.NewAgent(
		ai.WithCaller(caller),
	)

	if _, err := io.Copy(agent2, msg); err != nil {
		errnie.Error(err)
		return nil
	}

	agent3 := ai.NewAgent(
		ai.WithCaller(caller),
	)

	if _, err := io.Copy(agent3, msg); err != nil {
		errnie.Error(err)
		return nil
	}

	registry := system.NewRegistry(
		system.WithComponent(system.ComponentTypeAgent, agent1),
		system.WithComponent(system.ComponentTypeAgent, agent2),
		system.WithComponent(system.ComponentTypeAgent, agent3),
	)

	return &MultiAgent{
		wf: workflow.NewGraph(
			workflow.WithRegistry(registry),

			workflow.WithNode(
				&workflow.Node{
					ID: agent1.Name,
					Component: workflow.NewPipeline(
						agent1,
						workflow.NewFeedback(provider.NewOpenAIProvider(), agent1),
						workflow.NewConverter(),
					),
				},
			),
			workflow.WithNode(
				&workflow.Node{
					ID: agent2.Name,
					Component: workflow.NewPipeline(
						agent2,
						workflow.NewFeedback(provider.NewOpenAIProvider(), agent2),
						workflow.NewConverter(),
					),
				},
			),
			workflow.WithNode(
				&workflow.Node{
					ID: agent3.Name,
					Component: workflow.NewPipeline(
						agent3,
						workflow.NewFeedback(provider.NewOpenAIProvider(), agent3),
						workflow.NewConverter(),
					),
				},
			),
			workflow.WithEdge(
				&workflow.Edge{
					From: agent1.Name,
					To:   agent2.Name,
				},
			),
			workflow.WithEdge(
				&workflow.Edge{
					From: agent1.Name,
					To:   agent3.Name,
				},
			),
			workflow.WithEdge(
				&workflow.Edge{
					From: agent2.Name,
					To:   agent3.Name,
				},
			),
			workflow.WithEdge(
				&workflow.Edge{
					From: agent3.Name,
					To:   agent1.Name,
				},
			),
		),
	}
}

func (multiagent *MultiAgent) Run() error {
	errnie.Debug("multiagent.Run")

	if _, err := io.Copy(os.Stdout, multiagent.wf); err != nil {
		errnie.Error(err)
		return nil
	}

	return nil
}

func (multiagent *MultiAgent) Read(p []byte) (n int, err error) {
	errnie.Debug("multiagent.Read")
	return multiagent.wf.Read(p)
}

func (multiagent *MultiAgent) Write(p []byte) (n int, err error) {
	errnie.Debug("multiagent.Write")
	return multiagent.wf.Write(p)
}

func (multiagent *MultiAgent) Close() error {
	errnie.Debug("multiagent.Close")
	return multiagent.wf.Close()
}
