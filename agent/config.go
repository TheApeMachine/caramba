package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/caramba/tweaker"
)

/*
Config defines the configuration for an agent, including its identity,
behavior settings, and capabilities. It holds the agent's name, role,
system prompt, conversation thread, available tools, and generation
parameters.
*/
type Config struct {
	Name         string           // Unique identifier for the agent
	Role         string           // The role or purpose of the agent
	SystemPrompt string           // Base system prompt for the agent
	Thread       *provider.Thread // Conversation history and context
	Toolset      *tools.Toolset   // Set of tools available to the agent
	Temperature  float32          // Randomness in response generation
}

/*
NewConfig creates and returns a new agent configuration with the specified
parameters. It initializes a new conversation thread with the system prompt
and sets up the agent's toolset.

Parameters:

	system: The system prompt that defines the agent's base behavior
	role: The role or purpose assigned to the agent
	name: A unique identifier for the agent
	toolset: The set of tools available to the agent

Returns:

	*Config: A new configuration instance initialized with the provided parameters
*/
func NewConfig(system, role, name string, toolset *tools.Toolset) *Config {
	return &Config{
		Name:         name,
		Role:         role,
		SystemPrompt: system,
		Thread: provider.NewThread(
			provider.NewMessage(provider.RoleSystem,
				tweaker.GetSystemPrompt(
					system, name, role, toolset.String(),
				),
			),
		),
		Toolset:     toolset,
		Temperature: 0.1,
	}
}
