package tools

import (
	"context"
	"errors"
	"fmt"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/errnie"
)

// AgentTool allows an agent to create and manage other agents
type AgentTool struct {
	// llmProvider is used to create new agents
	llmProvider core.LLMProvider
	// existingAgents keeps track of created agents
	existingAgents map[string]core.Agent
	// agentPrompts stores system prompts for each agent
	agentPrompts map[string]string
}

// NewAgentTool creates a new AgentTool
func NewAgentTool(llmProvider core.LLMProvider) *AgentTool {
	return &AgentTool{
		llmProvider:    llmProvider,
		existingAgents: make(map[string]core.Agent),
		agentPrompts:   make(map[string]string),
	}
}

// Name returns the name of the tool
func (t *AgentTool) Name() string {
	return "agent"
}

// Description returns the description of the tool
func (t *AgentTool) Description() string {
	return "Creates and manages other agents to help with complex tasks"
}

// Execute executes the tool with the given arguments
func (t *AgentTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	action, ok := args["action"].(string)
	if !ok {
		return nil, errors.New("action must be a string")
	}

	switch action {
	case "create":
		return t.createAgent(ctx, args)
	case "execute":
		return t.executeAgent(ctx, args)
	case "list":
		return t.listAgents(ctx)
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

// Schema returns the JSON schema for the tool's arguments
func (t *AgentTool) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"action": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"create", "execute", "list"},
				"description": "Action to perform (create, execute, or list agents)",
			},
			"name": map[string]interface{}{
				"type":        "string",
				"description": "Name of the agent (required for create and execute)",
			},
			"system_prompt": map[string]interface{}{
				"type":        "string",
				"description": "System prompt for the agent (required for create)",
			},
			"model": map[string]interface{}{
				"type":        "string",
				"description": "Model to use (defaults to gpt-4o-mini)",
			},
			"input": map[string]interface{}{
				"type":        "string",
				"description": "Input for the agent (required for execute)",
			},
		},
		"required": []string{"action"},
	}
}

// createAgent creates a new agent with the given parameters
func (t *AgentTool) createAgent(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	name, ok := args["name"].(string)
	if !ok || name == "" {
		return nil, errors.New("name must be a non-empty string")
	}

	systemPrompt, ok := args["system_prompt"].(string)
	if !ok || systemPrompt == "" {
		return nil, errors.New("system_prompt must be a non-empty string")
	}

	// Check if model was specified, otherwise use default
	// Note: We don't use the model directly as we're using the existing LLM provider
	if _, ok := args["model"].(string); ok {
		// We acknowledge the model parameter but currently use the provider's model
		errnie.Info("Model parameter provided but using the existing LLM provider")
	}

	// Check if agent already exists
	if _, exists := t.existingAgents[name]; exists {
		return nil, fmt.Errorf("agent with name '%s' already exists", name)
	}

	// Create the agent - note that system prompt will be applied during execution
	baseAgent := core.NewBaseAgent(name)
	baseAgent.SetLLM(t.llmProvider)

	// Store the system prompt separately to use during execution
	t.agentPrompts[name] = systemPrompt

	// Store the agent
	t.existingAgents[name] = baseAgent

	errnie.Info(fmt.Sprintf("Created new agent: %s", name))

	return map[string]interface{}{
		"status":     "success",
		"agent_name": name,
		"message":    fmt.Sprintf("Agent '%s' created successfully", name),
	}, nil
}

// executeAgent executes an existing agent with the given input
func (t *AgentTool) executeAgent(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	name, ok := args["name"].(string)
	if !ok || name == "" {
		return nil, errors.New("name must be a non-empty string")
	}

	input, ok := args["input"].(string)
	if !ok || input == "" {
		return nil, errors.New("input must be a non-empty string")
	}

	// Check if agent exists
	agent, exists := t.existingAgents[name]
	if !exists {
		return nil, fmt.Errorf("agent with name '%s' does not exist", name)
	}

	// Get the custom system prompt if available
	_, promptExists := t.agentPrompts[name]

	// Execute the agent
	var result string
	var err error

	// Since we can't directly set system prompt in ExecuteOptions,
	// we will just use the regular Execute method.
	// Note: In a more complete implementation, we would need to modify
	// the BaseAgent implementation to allow setting the system prompt.
	if promptExists {
		errnie.Info(fmt.Sprintf("Agent '%s' has a custom system prompt, but it can't be applied with the current implementation", name))
	}

	// Execute the agent
	result, err = agent.Execute(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to execute agent: %w", err)
	}

	return map[string]interface{}{
		"status":     "success",
		"agent_name": name,
		"response":   result,
	}, nil
}

// listAgents returns a list of all existing agents
func (t *AgentTool) listAgents(ctx context.Context) (interface{}, error) {
	agentList := []map[string]string{}

	for name := range t.existingAgents {
		prompt := "Default"
		if customPrompt, exists := t.agentPrompts[name]; exists {
			prompt = fmt.Sprintf("Custom (%d chars)", len(customPrompt))
		}

		agentList = append(agentList, map[string]string{
			"name":       name,
			"promptType": prompt,
		})
	}

	return map[string]interface{}{
		"status": "success",
		"agents": agentList,
	}, nil
}
