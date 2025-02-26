package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/theapemachine/errnie"
)

// BaseAgent provides a base implementation of the Agent interface
type BaseAgent struct {
	Name           string
	LLM            LLMProvider
	Memory         Memory
	Params         LLMParams
	Tools          []Tool
	Planner        Planner
	Messenger      Messenger
	toolsMu        sync.RWMutex
	IterationLimit int
}

// NewBaseAgent creates a new BaseAgent
func NewBaseAgent(name string) *BaseAgent {
	agent := &BaseAgent{
		Name:  name,
		Tools: make([]Tool, 0),
		Params: LLMParams{
			Messages:                  make([]LLMMessage, 0),
			Model:                     "gpt-4o-mini",
			Temperature:               0.7,
			MaxTokens:                 1024,
			Tools:                     make([]Tool, 0),
			ResponseFormatName:        "",
			ResponseFormatDescription: "",
			Schema:                    nil,
		},
		IterationLimit: 1,
	}
	// Create a default messenger for the agent
	agent.Messenger = NewInMemoryMessenger(name)
	return agent
}

// Execute runs the agent with the provided input and returns a response
func (a *BaseAgent) Execute(ctx context.Context, message LLMMessage) (string, error) {
	if a.LLM == nil {
		return "", errors.New("no LLM provider set")
	}

	a.injectMemories(ctx, message)

	// plan, err := a.createPlan(ctx, message)

	var response strings.Builder

	iteration := 0
	iterMsg := LLMMessage{
		Role:    message.Role,
		Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, message.Content),
	}

	for iteration < a.IterationLimit {
		a.Params.Messages = append(a.Params.Messages, iterMsg)

		res := a.LLM.GenerateResponse(ctx, a.Params)

		if res.Error != nil {
			return "", res.Error
		}

		response.WriteString(res.Content)

		iteration++

		iterMsg = LLMMessage{
			Role:    "assistant",
			Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, response.String()),
		}
	}

	var contextWindow strings.Builder

	for _, message := range a.Params.Messages {
		switch message.Role {
		case "user":
			contextWindow.WriteString(fmt.Sprintf("User: %s\n", message.Content))
		case "assistant":
			contextWindow.WriteString(fmt.Sprintf("Assistant: %s\n", message.Content))
		}
	}

	a.extractMemories(ctx, contextWindow.String())

	return response.String(), nil
}

func (a *BaseAgent) createPlan(ctx context.Context, message LLMMessage) (Plan, error) {
	if a.Planner != nil {
		plan, err := a.Planner.CreatePlan(ctx, message.Content)
		if err != nil {
			return Plan{}, fmt.Errorf("failed to create plan: %w", err)
		}

		return plan, nil
	}

	return Plan{}, nil
}

func (a *BaseAgent) injectMemories(ctx context.Context, message LLMMessage) {
	if a.Memory != nil {
		// Check if the memory system supports memory preparation
		if memoryEnhancer, ok := a.Memory.(MemoryEnhancer); ok {
			enhancedContext, err := memoryEnhancer.PrepareContext(ctx, a.Name, message.Content)
			if err == nil && enhancedContext != "" {
				message.Content = enhancedContext
				errnie.Info(fmt.Sprintf("Enhanced input with %d characters of memories", len(enhancedContext)-len(message.Content)))
			}
		}
	}
}

func (a *BaseAgent) extractMemories(ctx context.Context, contextWindow string) {
	if a.Memory != nil {
		if memoryExtractor, ok := a.Memory.(MemoryExtractor); ok {
			_, err := memoryExtractor.ExtractMemories(ctx, a.Name, contextWindow, "conversation")

			if err != nil {
				errnie.Error(err)
			}
		}
	}
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

		// Execute the tool
		for _, tool := range a.Tools {
			if tool.Name() == toolName {
				result, err := tool.Execute(ctx, args)

				if err != nil {
					return nil, err
				}

				toolResults[toolName] = result
			}
		}
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
	// Try to parse as JSON first (our providers return JSON for tool calls)
	var toolCalls []ToolCall
	err := json.Unmarshal([]byte(response), &toolCalls)
	if err == nil && len(toolCalls) > 0 {
		return toolCalls
	}

	// If not JSON array, check if it might be a single tool call
	var singleToolCall ToolCall
	err = json.Unmarshal([]byte(response), &singleToolCall)
	if err == nil && singleToolCall.Name != "" {
		return []ToolCall{singleToolCall}
	}

	// Fallback: parse text for tool calls using regex (simplified version)
	// In a production system, you would implement more robust parsing here
	// This is a basic implementation that looks for patterns like:
	// tool_name({"param1": "value1"})

	// Empty implementation - future enhancement
	return []ToolCall{}
}

// AddTool adds a new tool to the agent
func (a *BaseAgent) AddTool(tool Tool) error {
	a.toolsMu.Lock()
	defer a.toolsMu.Unlock()

	for _, existingTool := range a.Tools {
		if existingTool.Name() == tool.Name() {
			return fmt.Errorf("tool %s already exists", tool.Name())
		}
	}

	a.Tools = append(a.Tools, tool)
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

// SetSystemPrompt sets the system prompt for the agent
func (a *BaseAgent) SetSystemPrompt(prompt string) {
	a.Params.Messages = append(a.Params.Messages, SystemMessage(prompt))
}

// SetModel sets the model for the agent
func (a *BaseAgent) SetModel(model string) {
	a.Params.Model = model
}

// SetTemperature sets the temperature for the agent
func (a *BaseAgent) SetTemperature(temperature float64) {
	a.Params.Temperature = temperature
}

// SetIterationLimit sets the iteration limit for the agent
func (a *BaseAgent) SetIterationLimit(limit int) {
	a.IterationLimit = limit
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
		res := a.LLM.GenerateResponse(ctx, LLMParams{
			Messages: []LLMMessage{
				{
					Role:    "system",
					Content: prompt,
				},
			},
		})
		if res.Error != nil {
			return "", res.Error
		}

		// Optionally, send a response back to the sender
		if a.Messenger != nil {
			_, _ = a.Messenger.SendDirect(ctx, message.Sender, res.Content, MessageTypeText, nil)
		}

		return res.Content, nil
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
