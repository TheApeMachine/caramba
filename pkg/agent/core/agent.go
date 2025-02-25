package core

import (
	"context"
)

/*
Agent represents the interface for an LLM-powered agent.
It defines the core functionality for executing tasks, managing tools,
memory, planning, and communication with other agents.
*/
type Agent interface {
	// Execute runs the agent with the provided input and returns a response
	Execute(ctx context.Context, input string) (string, error)

	// ExecuteWithOptions runs the agent with additional options
	ExecuteWithOptions(ctx context.Context, input string, opts ...ExecuteOption) (string, error)

	// ExecuteWithIteration runs the agent using the iteration loop for self-improvement
	ExecuteWithIteration(ctx context.Context, input string, iterOptions *IterationOptions) (string, error)

	// AddTool adds a new tool to the agent
	AddTool(tool Tool) error

	// SetMemory sets the memory system for the agent
	SetMemory(memory Memory)

	// SetLLM sets the LLM provider for the agent
	SetLLM(llm LLMProvider)

	// SetPlanner sets the planner for the agent
	SetPlanner(planner Planner)

	// RunWorkflow executes a predefined workflow
	RunWorkflow(ctx context.Context, workflow Workflow, input map[string]interface{}) (map[string]interface{}, error)

	// GetMessenger returns the agent's messenger
	GetMessenger() Messenger

	// SetMessenger sets the agent's messenger
	SetMessenger(messenger Messenger)

	// HandleMessage processes an incoming message
	HandleMessage(ctx context.Context, message Message) (string, error)
}

/*
ExecuteOption represents an option for executing an agent.
It follows the functional options pattern for configuring agent execution.
*/
type ExecuteOption func(*ExecuteOptions)

/*
ExecuteOptions contains all the options for executing an agent.
These options control various aspects of the agent's execution behavior.
*/
type ExecuteOptions struct {
	/* MaxTokens limits the number of tokens in the generated response */
	MaxTokens int
	/* Temperature controls randomness in generation (higher = more random) */
	Temperature float64
	/* StreamHandler is a callback function that receives chunks of streaming text */
	StreamHandler func(string)
}

/*
WithMaxTokens sets the maximum number of tokens for the execution.
This helps control the length of the generated response.

Parameters:
  - maxTokens: The maximum number of tokens to generate

Returns:
  - An ExecuteOption that can be passed to ExecuteWithOptions
*/
func WithMaxTokens(maxTokens int) ExecuteOption {
	return func(o *ExecuteOptions) {
		o.MaxTokens = maxTokens
	}
}

/*
WithTemperature sets the temperature for the execution.
Temperature controls the randomness of the generated response.

Parameters:
  - temperature: A value typically between 0 and 1, where higher values produce more random outputs

Returns:
  - An ExecuteOption that can be passed to ExecuteWithOptions
*/
func WithTemperature(temperature float64) ExecuteOption {
	return func(o *ExecuteOptions) {
		o.Temperature = temperature
	}
}

/*
WithStreamHandler sets a handler for streaming responses.
This allows for processing partial responses as they are generated.

Parameters:
  - handler: A function that receives chunks of text as they are generated

Returns:
  - An ExecuteOption that can be passed to ExecuteWithOptions
*/
func WithStreamHandler(handler func(string)) ExecuteOption {
	return func(o *ExecuteOptions) {
		o.StreamHandler = handler
	}
}
