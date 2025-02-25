package core

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/theapemachine/errnie"
)

// BaseAgent provides a base implementation of the Agent interface
type BaseAgent struct {
	Name      string
	LLM       LLMProvider
	Memory    Memory
	Tools     map[string]Tool
	Planner   Planner
	Messenger Messenger
	toolsMu   sync.RWMutex
}

// NewBaseAgent creates a new BaseAgent
func NewBaseAgent(name string) *BaseAgent {
	agent := &BaseAgent{
		Name:  name,
		Tools: make(map[string]Tool),
	}
	// Create a default messenger for the agent
	agent.Messenger = NewInMemoryMessenger(name)
	return agent
}

// Execute runs the agent with the provided input and returns a response
func (a *BaseAgent) Execute(ctx context.Context, input string) (string, error) {
	return a.ExecuteWithOptions(ctx, input)
}

// ExecuteWithOptions runs the agent with additional options
func (a *BaseAgent) ExecuteWithOptions(ctx context.Context, input string, opts ...ExecuteOption) (string, error) {
	if a.LLM == nil {
		return "", errors.New("no LLM provider set")
	}

	options := &ExecuteOptions{
		MaxTokens:   1024,
		Temperature: 0.7,
	}

	for _, opt := range opts {
		opt(options)
	}

	// If we have a planner, use it to create and execute a plan
	if a.Planner != nil {
		plan, err := a.Planner.CreatePlan(ctx, input)
		if err != nil {
			return "", fmt.Errorf("failed to create plan: %w", err)
		}

		return a.Planner.ExecutePlan(ctx, plan)
	}

	// Otherwise, just send the input to the LLM
	llmOptions := LLMOptions{
		MaxTokens:    options.MaxTokens,
		Temperature:  options.Temperature,
		SystemPrompt: fmt.Sprintf("You are %s, an AI assistant. Answer the question to the best of your abilities.", a.Name),
	}

	if options.StreamHandler != nil {
		err := a.LLM.StreamResponse(ctx, input, llmOptions, options.StreamHandler)
		return "", err // We don't return the response when streaming
	}

	return a.LLM.GenerateResponse(ctx, input, llmOptions)
}

// ExecuteWithIteration runs the agent using the iteration loop for self-improvement
func (a *BaseAgent) ExecuteWithIteration(ctx context.Context, input string, iterOptions *IterationOptions) (string, error) {
	if a.LLM == nil {
		return "", errors.New("no LLM provider set")
	}

	// Create an iterator with the provided options
	iterator := NewIterator(iterOptions)

	// Run the iteration process
	return iterator.Run(ctx, a, input)
}

// GetToolResults processes the results of tool calls for the iterator
func (a *BaseAgent) GetToolResults(ctx context.Context, response string) (map[string]interface{}, error) {
	// This is a simplified implementation
	// A real implementation would parse the response for tool calls, execute them, and return results

	toolResults := make(map[string]interface{})

	// Extract tool calls from the response
	// This is a placeholder - real implementation would depend on the format of tool calls
	toolCalls := extractToolCalls(response)

	// Execute each tool call
	for _, toolCall := range toolCalls {
		toolName := toolCall.Name
		args := toolCall.Args

		// Find the tool
		a.toolsMu.RLock()
		tool, exists := a.Tools[toolName]
		a.toolsMu.RUnlock()

		if !exists {
			continue // Skip tools that don't exist
		}

		// Execute the tool
		result, err := tool.Execute(ctx, args)
		if err != nil {
			// Log the error but continue with other tools
			continue
		}

		// Store the result
		toolResults[toolName] = result
	}

	return toolResults, nil
}

// ToolCall represents a parsed tool call from the agent's response
type ToolCall struct {
	Name string
	Args map[string]interface{}
}

// extractToolCalls parses tool calls from the agent's response
// This is a placeholder implementation - real implementation would depend on the format
func extractToolCalls(response string) []ToolCall {
	// This is a simplified implementation
	// A real implementation would parse the response for tool calls
	return []ToolCall{}
}

// AddTool adds a new tool to the agent
func (a *BaseAgent) AddTool(tool Tool) error {
	a.toolsMu.Lock()
	defer a.toolsMu.Unlock()

	if _, exists := a.Tools[tool.Name()]; exists {
		return fmt.Errorf("tool %s already exists", tool.Name())
	}

	a.Tools[tool.Name()] = tool
	return nil
}

// SetMemory sets the memory system for the agent
func (a *BaseAgent) SetMemory(memory Memory) {
	a.Memory = memory
}

// SetLLM sets the LLM provider for the agent
func (a *BaseAgent) SetLLM(llm LLMProvider) {
	a.LLM = llm
}

// SetPlanner sets the planner for the agent
func (a *BaseAgent) SetPlanner(planner Planner) {
	a.Planner = planner
}

// GetMessenger returns the agent's messenger
func (a *BaseAgent) GetMessenger() Messenger {
	return a.Messenger
}

// SetMessenger sets the agent's messenger
func (a *BaseAgent) SetMessenger(messenger Messenger) {
	a.Messenger = messenger
}

// HandleMessage processes an incoming message
func (a *BaseAgent) HandleMessage(ctx context.Context, message Message) (string, error) {
	// This is a basic implementation - specific agent types can override this
	if a.LLM == nil {
		return "", errors.New("no LLM provider set")
	}

	// For direct messages, we'll just pass the content to the LLM
	if message.IsDirect() {
		promptTemplate := "Message from %s: %s\nPlease respond to this message."
		prompt := fmt.Sprintf(promptTemplate, message.Sender, message.Content)

		// Use the agent's LLM to generate a response
		llmOptions := LLMOptions{
			MaxTokens:    1024,
			Temperature:  0.7,
			SystemPrompt: fmt.Sprintf("You are %s, an AI assistant. You are receiving a message from another agent.", a.Name),
		}

		response, err := a.LLM.GenerateResponse(ctx, prompt, llmOptions)
		if err != nil {
			return "", err
		}

		// Optionally, send a response back to the sender
		if a.Messenger != nil {
			_, _ = a.Messenger.SendDirect(ctx, message.Sender, response, MessageTypeText, nil)
		}

		return response, nil
	}

	// For topic messages or broadcasts, we might just log them
	msgType := "broadcast"
	if message.IsTopic() {
		msgType = fmt.Sprintf("topic %s", message.Topic)
	}

	logMsg := fmt.Sprintf("Agent %s received %s message from %s: %s",
		a.Name, msgType, message.Sender, message.Content)

	errnie.Info(logMsg)
	return "", nil
}

// RunWorkflow executes a predefined workflow
func (a *BaseAgent) RunWorkflow(ctx context.Context, workflow Workflow, input map[string]interface{}) (map[string]interface{}, error) {
	if workflow == nil {
		return nil, errors.New("workflow is nil")
	}

	return workflow.Execute(ctx, input)
}
