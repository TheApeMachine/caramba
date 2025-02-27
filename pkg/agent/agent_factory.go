/*
Package agent provides the core agent functionality for the Caramba framework.
It includes agent factories, tools, memory management, and LLM provider integrations.
This package allows for creating and configuring various types of specialized agents
with different capabilities and integrations.
*/
package agent

import (
	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/llm"
	"github.com/theapemachine/caramba/pkg/agent/memory"
)

/*
AgentFactory provides methods to create pre-configured agents with different capabilities.
It serves as a factory pattern implementation for creating various types of agents with
appropriate tools and configurations.
*/
type AgentFactory struct{}

/*
NewAgentFactory creates a new agent factory instance.
Returns a pointer to the initialized AgentFactory.
*/
func NewAgentFactory() *AgentFactory {
	return &AgentFactory{}
}

/*
CreateBasicAgent creates a basic agent with the default configuration.
It uses the configuration to determine which LLM provider to use.

Parameters:
  - name: The name to identify the agent
  - apiKey: The API key for the LLM provider

Returns:
  - A configured core.Agent instance
*/
func (f *AgentFactory) CreateBasicAgent(name string, apiKey string) core.Agent {
	// Determine which provider to use from configuration
	var llmProvider core.LLMProvider

	if viper.GetBool("providers.openai") {
		model := viper.GetString("models.openai")
		llmProvider = llm.NewOpenAIProvider(apiKey, model)
	} else if viper.GetBool("providers.anthropic") {
		model := viper.GetString("models.anthropic")
		llmProvider = llm.NewAnthropicProvider(apiKey, model)
	} else {
		// Default to OpenAI with a default model
		llmProvider = llm.NewOpenAIProvider(apiKey, "gpt-4o-mini")
	}

	// Create the agent
	return core.NewAgentBuilder(name).
		WithLLM(llmProvider).
		WithMemory(memory.NewInMemoryStore()).
		Build()
}

/*
AgentConfig provides configuration for creating a custom agent.
It holds all the components needed to construct a fully customized agent instance.
*/
type AgentConfig struct {
	/* Name is the identifier for the agent */
	Name string
	/* LLMProvider is the language model provider for the agent */
	LLMProvider core.LLMProvider
	/* Memory is the storage mechanism for the agent's knowledge and context */
	Memory core.Memory
	/* Planner is the component responsible for task planning and execution */
	Planner core.Planner
	/* Tools is a collection of capabilities the agent can use */
	Tools []core.Tool
}
