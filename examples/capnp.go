package examples

import (
	"github.com/theapemachine/caramba/pkg/api/ai"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

// CapnpExample demonstrates using Cap'n Proto interfaces with an agent
type CapnpExample struct {
	agents []*ai.Agent
}

// NewCapnp creates a new Cap'n Proto example with a single agent
func NewCapnp() *CapnpExample {
	errnie.Debug("examples.NewCapnp")

	agents := make([]*ai.Agent, 0)

	for _, agentName := range []string{"agent1", "agent2", "agent3"} {
		agent, err := ai.NewCapnpAgent(agentName)

		if err != nil {
			errnie.Error(err)
			return nil
		}

		agent.SetContext(*agent.AddTool("inspect"))
		agent.SetContext(*agent.AddTool("message"))
		agent.SetContext(*agent.AddTool("optimize"))

		agent.SetContext(*agent.AddMessage(
			"system",
			"",
			tweaker.GetSystemPrompt("default"),
		))

		agent.SetContext(*agent.AddMessage(
			"user",
			"danny",
			"Try to find other agents in the system, then have a conversation with them.\n\nYour name is: "+agentName+", so you don't have to message yourself.\n\nYou can use the optimize tool to optimize your performance and behavior.\n\nRemember: you can make multiple tool calls in a single response, so always keep evaluating whether you should change values to adjust to the current situation.",
		))

		agents = append(agents, agent)
	}

	ai.Agents = agents

	return &CapnpExample{
		agents: agents,
	}
}

func (example *CapnpExample) Run() (err error) {
	errnie.Info("Starting Cap'n Proto example")

	for {
		for _, agent := range example.agents {
			agent.SetContext(*agent.Ask())
		}
	}
}
