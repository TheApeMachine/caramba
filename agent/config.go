package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tweaker"
	"github.com/theapemachine/errnie"
)

/*
Config defines the configuration for an agent, including its identity,
behavior settings, and capabilities. It holds the agent's name, role,
system prompt, conversation thread, available tools, and generation
parameters.
*/
type Config struct {
	name         string           // Unique identifier for the agent
	role         string           // The role or purpose of the agent
	systemPrompt string           // Base system prompt for the agent
	thread       *provider.Thread // Conversation history and context
	temperature  float32          // Randomness in response generation
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
func NewConfig(system, role, name, toolschemas string) *Config {
	errnie.Debug("new config", "name", name, "role", role)

	return &Config{
		name:         name,
		role:         role,
		systemPrompt: system,
		thread: provider.NewThread(
			provider.NewMessage(provider.RoleSystem,
				tweaker.GetSystemPrompt(
					system, name, role, toolschemas,
				),
			),
		),
		temperature: 0.1,
	}
}

func (config *Config) Name() string {
	return config.name
}

func (config *Config) Role() string {
	return config.role
}

func (config *Config) SystemPrompt() string {
	return config.systemPrompt
}

func (config *Config) Thread() *provider.Thread {
	return config.thread
}

func (config *Config) Temperature() float32 {
	return config.temperature
}
