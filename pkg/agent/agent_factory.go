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
	"github.com/theapemachine/caramba/pkg/agent/tools"
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
CreateResearchAgent creates an agent optimized for research tasks.
It builds upon the basic agent and adds research-specific tools like web search and calculator.

Parameters:
  - name: The name to identify the agent
  - apiKey: The API key for the LLM provider
  - searchAPIKey: The API key for the search service
  - searchID: The search engine ID

Returns:
  - A configured core.Agent instance with research capabilities
*/
func (f *AgentFactory) CreateResearchAgent(name string, apiKey string, searchAPIKey string, searchID string) core.Agent {
	// Create a basic agent first
	agent := f.CreateBasicAgent(name, apiKey)

	// Add research-specific tools
	webSearch := tools.NewWebSearch(searchAPIKey, searchID)
	calculator := tools.NewCalculator()

	// Add the tools to the agent
	agent.AddTool(webSearch)
	agent.AddTool(calculator)

	return agent
}

/*
CreateBrowserAgent creates an agent with browser capabilities.
This agent can navigate websites, render content, and interact with web pages.

Parameters:
  - name: The name to identify the agent
  - apiKey: The API key for the LLM provider
  - browserlessURL: The URL endpoint for the browserless service
  - browserlessToken: The authentication token for the browserless service

Returns:
  - A configured core.Agent instance with browser capabilities
*/
func (f *AgentFactory) CreateBrowserAgent(name string, apiKey string, browserlessURL string, browserlessToken string) core.Agent {
	// Create a basic agent first
	agent := f.CreateBasicAgent(name, apiKey)

	// Add browser-specific tools
	browserTool := tools.NewBrowserTool(browserlessURL, browserlessToken)

	// Add the tool to the agent
	agent.AddTool(browserTool)

	return agent
}

/*
CreateGitHubAgent creates an agent with GitHub integration capabilities.
This agent can interact with GitHub repositories, issues, and pull requests.

Parameters:
  - name: The name to identify the agent
  - apiKey: The API key for the LLM provider
  - githubToken: The GitHub authentication token
  - defaultOwner: The default GitHub repository owner/organization
  - defaultRepo: The default GitHub repository name

Returns:
  - A configured core.Agent instance with GitHub integration
*/
func (f *AgentFactory) CreateGitHubAgent(name string, apiKey string, githubToken string, defaultOwner string, defaultRepo string) core.Agent {
	// Create a basic agent first
	agent := f.CreateBasicAgent(name, apiKey)

	// Add GitHub-specific tools
	githubTool := tools.NewGitHubTool(githubToken, defaultOwner, defaultRepo)

	// Add the tool to the agent
	agent.AddTool(githubTool)

	return agent
}

/*
CreateAzureDevOpsAgent creates an agent with Azure DevOps integration capabilities.
This agent can work with Azure DevOps boards, work items, and project management features.

Parameters:
  - name: The name to identify the agent
  - apiKey: The API key for the LLM provider
  - organization: The Azure DevOps organization name
  - pat: The Personal Access Token for Azure DevOps
  - defaultProject: The default Azure DevOps project

Returns:
  - A configured core.Agent instance with Azure DevOps integration
*/
func (f *AgentFactory) CreateAzureDevOpsAgent(name string, apiKey string, organization string, pat string, defaultProject string) core.Agent {
	// Create a basic agent first
	agent := f.CreateBasicAgent(name, apiKey)

	// Add Azure DevOps-specific tools
	azureDevOpsTool := tools.NewAzureDevOpsTool(organization, pat, defaultProject)

	// Add the tool to the agent
	agent.AddTool(azureDevOpsTool)

	return agent
}

/*
CreateSlackAgent creates an agent with Slack integration capabilities.
This agent can send and receive messages in Slack channels and interact with users.

Parameters:
  - name: The name to identify the agent
  - apiKey: The API key for the LLM provider
  - slackToken: The Slack API token
  - defaultChannel: The default Slack channel to post messages to

Returns:
  - A configured core.Agent instance with Slack integration
*/
func (f *AgentFactory) CreateSlackAgent(name string, apiKey string, slackToken string, defaultChannel string) core.Agent {
	// Create a basic agent first
	agent := f.CreateBasicAgent(name, apiKey)

	// Add Slack-specific tools
	slackTool := tools.NewSlackTool(slackToken, defaultChannel)

	// Add the tool to the agent
	agent.AddTool(slackTool)

	return agent
}

/*
CreateCustomAgent creates a fully customized agent based on the provided configuration.
This method offers maximum flexibility for creating specialized agents with custom components.

Parameters:
  - config: The AgentConfig struct containing all configuration options

Returns:
  - A configured core.Agent instance according to the provided configuration
*/
func (f *AgentFactory) CreateCustomAgent(config AgentConfig) core.Agent {
	builder := core.NewAgentBuilder(config.Name)

	// Set LLM provider
	if config.LLMProvider != nil {
		builder.WithLLM(config.LLMProvider)
	}

	// Set memory
	if config.Memory != nil {
		builder.WithMemory(config.Memory)
	} else {
		builder.WithMemory(memory.NewInMemoryStore())
	}

	// Set planner
	if config.Planner != nil {
		builder.WithPlanner(config.Planner)
	}

	// Add tools
	for _, tool := range config.Tools {
		builder.WithTool(tool)
	}

	return builder.Build()
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
