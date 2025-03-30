package examples

import (
	"context"
	"fmt"
	"os"

	"github.com/theapemachine/caramba/pkg/api/ai"
	"github.com/theapemachine/caramba/pkg/api/provider"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/tweaker"
)

// CapnpExample demonstrates using Cap'n Proto interfaces with multiple agents
type CapnpExample struct {
	agents   map[string]ai.Agent
	contexts map[string]*provider.ProviderParams
}

// NewCapnp creates a new Cap'n Proto example with multiple agents
func NewCapnp() *CapnpExample {
	errnie.Debug("examples.NewCapnp")

	prvdr := provider.NewProvider(os.Getenv("OPENAI_API_KEY"))

	agents := make(map[string]ai.Agent)
	contexts := make(map[string]*provider.ProviderParams)

	// Create three agents and their contexts
	for i := 1; i <= 3; i++ {
		agentName := fmt.Sprintf("agent%d", i)
		agent, err := ai.NewAgent(prvdr)

		if errnie.Error(err) != nil {
			return nil
		}

		context, err := provider.NewConversation()

		if errnie.Error(err) != nil {
			return nil
		}

		if err = provider.AddTool(context, "system"); errnie.Error(err) != nil {
			return nil
		}

		if err = provider.AddSystemMessage(
			context,
			tweaker.GetSystemPrompt("default"),
		); errnie.Error(err) != nil {
			return nil
		}

		agents[agentName] = agent
		contexts[agentName] = context
	}

	return &CapnpExample{
		agents:   agents,
		contexts: contexts,
	}
}

func (example *CapnpExample) Run() (err error) {
	errnie.Info("Starting Cap'n Proto multi-agent example")

	for name, agent := range example.agents {
		if example.contexts[name], err = ai.Ask(
			context.Background(),
			agent,
			example.contexts[name],
		); err != nil {
			return errnie.Error(err)
		}
	}

	return nil
}
