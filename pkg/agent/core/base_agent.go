package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/agent/util"
	"github.com/theapemachine/errnie"
)

// BaseAgent provides a base implementation of the Agent interface
type BaseAgent struct {
	Name      string
	LLM       LLMProvider
	Memory    Memory
	Tools     []Tool
	Planner   Planner
	Messenger Messenger
	toolsMu   sync.RWMutex
}

// NewBaseAgent creates a new BaseAgent
func NewBaseAgent(name string) *BaseAgent {
	agent := &BaseAgent{
		Name:  name,
		Tools: make([]Tool, 0),
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

	// If we have a memory system, use it to enhance the input with relevant memories
	enhancedInput := input
	if a.Memory != nil {
		// Check if the memory system supports memory preparation
		if memoryEnhancer, ok := a.Memory.(MemoryEnhancer); ok {
			enhancedContext, err := memoryEnhancer.PrepareContext(ctx, a.Name, input)
			if err == nil && enhancedContext != "" {
				enhancedInput = enhancedContext
				errnie.Info(fmt.Sprintf("Enhanced input with %d characters of memories", len(enhancedContext)-len(input)))
			}
		}
	}

	// If we have a planner, use it to create and execute a plan
	if a.Planner != nil {
		plan, err := a.Planner.CreatePlan(ctx, enhancedInput)
		if err != nil {
			return "", fmt.Errorf("failed to create plan: %w", err)
		}

		return a.Planner.ExecutePlan(ctx, plan)
	}

	var response strings.Builder

	// Otherwise, just send the input to the LLM
	params := LLMParams{
		Messages: []LLMMessage{
			{
				Role:    "system",
				Content: viper.GetViper().GetString("templates.planner"),
			},
		},
		Model:  "gpt-4o-mini",
		Tools:  a.Tools,
		Schema: util.GenerateSchema[Plan](),
	}

	if options.StreamHandler != nil {
		for chunk := range a.LLM.StreamResponse(ctx, LLMParams{
			Messages: []LLMMessage{
				{
					Role:    "system",
					Content: viper.GetViper().GetString("templates.planner"),
				},
				{
					Role:    "user",
					Content: enhancedInput,
				},
			},
		}) {
			response.WriteString(chunk.Content)
			options.StreamHandler(chunk.Content)
		}
	} else {
		res, err := a.LLM.GenerateResponse(ctx, params)
		if err != nil {
			return "", err
		}

		response.WriteString(res)
	}

	// Store the interaction in memory if available
	if a.Memory != nil {
		// Store the user input
		inputKey := fmt.Sprintf("user_input_%d", time.Now().UnixNano())
		_ = a.Memory.Store(ctx, inputKey, input)

		// Store the agent's response
		responseKey := fmt.Sprintf("agent_response_%d", time.Now().UnixNano())
		_ = a.Memory.Store(ctx, responseKey, response)

		// Extract memories if the memory system supports it
		if memoryExtractor, ok := a.Memory.(MemoryExtractor); ok {
			// Extract memories from both the input and response
			interaction := fmt.Sprintf("User: %s\n\nAgent: %s", input, response)
			memoryIDs, err := memoryExtractor.ExtractMemories(ctx, a.Name, interaction, "conversation")
			if err != nil {
				errnie.Error(err)
			} else if len(memoryIDs) > 0 {
				errnie.Info(fmt.Sprintf("Extracted %d memories from conversation", len(memoryIDs)))
			}
		}
	}

	return response.String(), nil
}

// ExecuteWithIteration runs the agent using the iteration loop for self-improvement
func (a *BaseAgent) ExecuteWithIteration(ctx context.Context, input string, iterOptions *IterationOptions) (string, error) {
	if a.LLM == nil {
		return "", errors.New("no LLM provider set")
	}

	// Create an iterator with the provided options
	iterator := NewIterator(iterOptions)

	// If we have a memory system, use it to enhance the input with relevant memories
	enhancedInput := input
	if a.Memory != nil {
		// Check if the memory system supports memory preparation
		if memoryEnhancer, ok := a.Memory.(MemoryEnhancer); ok {
			enhancedContext, err := memoryEnhancer.PrepareContext(ctx, a.Name, input)
			if err == nil && enhancedContext != "" {
				enhancedInput = enhancedContext
				errnie.Info("Enhanced input with memories for iteration")
			}
		}
	}

	// Run the iteration process with enhanced input
	response, err := iterator.Run(ctx, a, enhancedInput)
	if err != nil {
		return "", err
	}

	// After iteration completes, store the final result in memory
	if a.Memory != nil {
		// Store the user input
		inputKey := fmt.Sprintf("user_input_%d", time.Now().UnixNano())
		_ = a.Memory.Store(ctx, inputKey, input)

		// Store the agent's final response
		responseKey := fmt.Sprintf("agent_response_%d", time.Now().UnixNano())
		_ = a.Memory.Store(ctx, responseKey, response)

		// Extract memories if the memory system supports it
		if memoryExtractor, ok := a.Memory.(MemoryExtractor); ok {
			// Create a combined interaction text with the final response
			interaction := fmt.Sprintf("User: %s\n\nAgent (after iteration): %s", input, response)
			memoryIDs, err := memoryExtractor.ExtractMemories(ctx, a.Name, interaction, "conversation_iterated")
			if err != nil {
				errnie.Info(fmt.Sprintf("Memory extraction after iteration encountered an error: %v", err))
			} else if len(memoryIDs) > 0 {
				errnie.Info(fmt.Sprintf("Extracted %d memories from iterated conversation", len(memoryIDs)))
			}
		}
	}

	return response, nil
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
		response, err := a.LLM.GenerateResponse(ctx, LLMParams{
			Messages: []LLMMessage{
				{
					Role:    "system",
					Content: prompt,
				},
			},
		})
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
