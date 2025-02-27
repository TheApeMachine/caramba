package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/fatih/color"
	"github.com/theapemachine/caramba/pkg/output"
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

	// Log the execution start with summarized input
	output.Action("agent", "execute", fmt.Sprintf("%s processing: %s", a.Name, output.Summarize(message.Content, 60)))

	// Check for memory integration and enhance context if available
	memorySpinner := output.StartSpinner("Checking memory for relevant context")
	enhancedMessage := a.injectMemories(ctx, message)
	message = enhancedMessage // Use the enhanced message
	output.StopSpinner(memorySpinner, "Memory retrieval complete")

	// Create a plan if a planner is available
	if a.Planner != nil {
		planSpinner := output.StartSpinner("Planning execution strategy")
		_, err := a.createPlan(ctx, message)
		if err != nil {
			output.StopSpinner(planSpinner, "")
			output.Error("Planning failed", err)
		} else {
			output.StopSpinner(planSpinner, "Execution plan created")
		}
	}

	// Prepare for iterations
	var response strings.Builder
	iteration := 0

	// Format the message for the first iteration
	iterMsg := LLMMessage{
		Role:    message.Role,
		Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, message.Content),
	}

	// Run the iterations
	for iteration < a.IterationLimit {
		// Add the current message to the conversation
		a.Params.Messages = append(a.Params.Messages, iterMsg)

		// IMPORTANT: Copy tools to LLM parameters before making the API call
		a.toolsMu.RLock()
		a.Params.Tools = a.Tools
		a.toolsMu.RUnlock()

		// Show thinking spinner
		thinkingSpinner := output.StartSpinner(fmt.Sprintf("Iteration %d/%d: Agent thinking", iteration+1, a.IterationLimit))

		// Generate the response
		res := a.LLM.GenerateResponse(ctx, a.Params)

		// Handle errors
		if res.Error != nil {
			output.StopSpinner(thinkingSpinner, "")
			output.Error("LLM response generation failed", res.Error)
			return "", res.Error
		}

		// Success - stop spinner with appropriate message
		if len(res.ToolCalls) > 0 {
			output.StopSpinner(thinkingSpinner, fmt.Sprintf("Agent is using %d tools", len(res.ToolCalls)))

			// Log each tool call
			for _, toolCall := range res.ToolCalls {
				output.Action("agent", "tool_call", toolCall.Name)
			}
		} else {
			output.StopSpinner(thinkingSpinner, "Agent completed thinking")
		}

		// Add the response to the accumulated response
		response.WriteString(res.Content)

		// Next iteration
		iteration++

		// Prepare the next message as assistant's response
		iterMsg = LLMMessage{
			Role:    "assistant",
			Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, response.String()),
		}
	}

	// Build context window for memory extraction
	var contextWindow strings.Builder
	for _, message := range a.Params.Messages {
		switch message.Role {
		case "user":
			contextWindow.WriteString(fmt.Sprintf("User: %s\n", message.Content))
		case "assistant":
			contextWindow.WriteString(fmt.Sprintf("Assistant: %s\n", message.Content))
		}
	}

	// Extract memories from the conversation
	memExtractSpinner := output.StartSpinner("Processing and storing memories")
	a.extractMemories(ctx, contextWindow.String())
	output.StopSpinner(memExtractSpinner, "New memories stored")

	// Final reporting
	output.Result(fmt.Sprintf("Agent %s completed execution (%d tokens)", a.Name, estimateTokens(response.String())))

	return response.String(), nil
}

// StreamExecute runs the agent with the provided input and streams the response in real-time
func (a *BaseAgent) StreamExecute(ctx context.Context, message LLMMessage) (string, error) {
	if a.LLM == nil {
		return "", errors.New("no LLM provider set")
	}

	// Log the execution start with summarized input
	output.Action("agent", "stream_execute", fmt.Sprintf("%s processing: %s", a.Name, output.Summarize(message.Content, 60)))

	// Check for memory integration and enhance context if available
	memorySpinner := output.StartSpinner("Checking memory for relevant context")
	enhancedMessage := a.injectMemories(ctx, message)
	message = enhancedMessage // Use the enhanced message
	output.StopSpinner(memorySpinner, "Memory retrieval complete")

	// Create a plan if a planner is available
	if a.Planner != nil {
		planSpinner := output.StartSpinner("Planning execution strategy")
		_, err := a.createPlan(ctx, message)
		if err != nil {
			output.StopSpinner(planSpinner, "")
			output.Error("Planning failed", err)
		} else {
			output.StopSpinner(planSpinner, "Execution plan created")
		}
	}

	// Prepare for iterations
	var fullResponse strings.Builder
	iteration := 0

	// Format the message for the first iteration
	iterMsg := LLMMessage{
		Role:    message.Role,
		Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, message.Content),
	}

	// Run the iterations
	for iteration < a.IterationLimit {
		// Add the current message to the conversation
		a.Params.Messages = append(a.Params.Messages, iterMsg)

		// IMPORTANT: Copy tools to LLM parameters before making the API call
		a.toolsMu.RLock()
		a.Params.Tools = a.Tools
		a.toolsMu.RUnlock()

		// Show thinking spinner for initial pause only
		thinkingSpinner := output.StartSpinner(fmt.Sprintf("Iteration %d/%d: Agent thinking", iteration+1, a.IterationLimit))

		// Start streaming instead of generating full response at once
		output.StopSpinner(thinkingSpinner, "Agent is responding in real-time:")

		// Print a marker to indicate streaming response start
		output.Info("Streaming response begins:")
		fmt.Println(strings.Repeat("-", 40))

		// Track if tool calls were found during streaming
		toolCallsFound := false
		var toolCalls []ToolCall

		// Stream the response
		streamChan := a.LLM.StreamResponse(ctx, a.Params)

		// Create a buffer to accumulate the current response
		var streamedResponse strings.Builder

		// Process each chunk from the stream
		for chunk := range streamChan {
			if chunk.Error != nil {
				output.Error("Streaming failed", chunk.Error)
				return fullResponse.String(), chunk.Error
			}

			if len(chunk.ToolCalls) > 0 {
				// We found tool calls - handle them
				toolCallsFound = true
				toolCalls = chunk.ToolCalls

				// Log the tool calls detected with more detail
				for _, toolCall := range chunk.ToolCalls {
					// Create a summary of args for better logging
					argsSummary := ""
					if action, ok := toolCall.Args["action"].(string); ok {
						argsSummary = fmt.Sprintf("action=%s", action)

						// For browser tool, add more details based on the action
						if toolCall.Name == "browser" {
							switch action {
							case "search":
								if query, ok := toolCall.Args["query"].(string); ok {
									argsSummary += fmt.Sprintf(" query=%q", output.Summarize(query, 30))
								}
							case "navigate":
								if url, ok := toolCall.Args["url"].(string); ok {
									argsSummary += fmt.Sprintf(" url=%q", output.Summarize(url, 30))
								}
							case "extract":
								url, urlOk := toolCall.Args["url"].(string)
								selector, selectorOk := toolCall.Args["selector"].(string)
								if urlOk && selectorOk {
									argsSummary += fmt.Sprintf(" url=%q selector=%q",
										output.Summarize(url, 20), output.Summarize(selector, 15))
								}
							}
						}
					}

					if argsSummary == "" {
						// Generate a generic summary if we couldn't extract specific fields
						argsJSON, _ := json.Marshal(toolCall.Args)
						argsSummary = output.Summarize(string(argsJSON), 50)
					}

					output.Action("agent", "tool_call", fmt.Sprintf("%s: %s", toolCall.Name, argsSummary))
				}
			} else if chunk.Content != "" {
				// Check if the content is JSON
				content := chunk.Content

				// Try to parse as JSON if it starts with a brace
				if strings.HasPrefix(strings.TrimSpace(content), "{") {
					var jsonData map[string]interface{}
					if err := json.Unmarshal([]byte(content), &jsonData); err == nil {
						// Extract content field if it exists
						if contentValue, ok := jsonData["content"]; ok && contentValue != nil {
							if strContent, ok := contentValue.(string); ok {
								content = strContent
							}
						}
					}
				}

				// Only process and display non-empty content
				if content != "" {
					// Format the content before printing for better readability
					formattedContent := formatStreamedContent(content)

					// Print the content chunk directly to show real-time streaming
					fmt.Print(formattedContent)
					streamedResponse.WriteString(content)
				}
			}
		}

		// Append the complete streamed response to the full response
		fullResponse.WriteString(streamedResponse.String())

		// Print a marker to indicate streaming response end
		fmt.Println()
		fmt.Println(strings.Repeat("-", 40))
		output.Info("Streaming response complete")

		// Process tool calls if any were found
		if toolCallsFound {
			output.Info(fmt.Sprintf("Processing %d tool calls", len(toolCalls)))

			// Process tool calls
			for _, toolCall := range toolCalls {
				toolName := toolCall.Name
				args := toolCall.Args

				// More detailed output about the specific tool call
				fmt.Println()
				fmt.Println(strings.Repeat("=", 40))
				output.Title(fmt.Sprintf("TOOL CALL: %s", strings.ToUpper(toolName)))

				// Format and display args for better visibility
				fmt.Println("Arguments:")
				prettyArgs, _ := json.MarshalIndent(args, "", "  ")
				fmt.Println(string(prettyArgs))

				output.Action("agent", "execute_tool", toolName)
				toolSpinner := output.StartSpinner(fmt.Sprintf("Executing tool: %s", toolName))

				// Find and execute the tool
				toolFound := false
				for _, tool := range a.Tools {
					if tool.Name() == toolName {
						toolFound = true

						// Execute the tool
						result, err := tool.Execute(ctx, args)

						if err != nil {
							output.StopSpinner(toolSpinner, "")
							output.Error(fmt.Sprintf("Tool %s execution failed", toolName), err)
							continue
						}

						output.StopSpinner(toolSpinner, fmt.Sprintf("Tool %s executed successfully", toolName))

						// Format the result for display
						fmt.Println()
						output.Title(fmt.Sprintf("TOOL RESULT: %s", strings.ToUpper(toolName)))

						// Pretty-print the result
						var resultOutput string
						if resultStr, ok := result.(string); ok {
							resultOutput = resultStr
						} else {
							prettyResult, _ := json.MarshalIndent(result, "", "  ")
							resultOutput = string(prettyResult)
						}

						// Format the tool result for display with a clear separator
						fmt.Println(resultOutput)
						fmt.Println(strings.Repeat("=", 40))
						fmt.Println()

						// Add the tool result to the response
						toolResultStr := fmt.Sprintf("\n\nTool Result (%s):\n%v\n\n", toolName, resultOutput)
						streamedResponse.WriteString(toolResultStr)
					}
				}

				if !toolFound {
					output.StopSpinner(toolSpinner, "")
					output.Error(fmt.Sprintf("Tool %s not found", toolName), nil)
				}
			}
		}

		// Next iteration
		iteration++

		// Prepare the next message as assistant's response
		iterMsg = LLMMessage{
			Role:    "assistant",
			Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, fullResponse.String()),
		}
	}

	// Build context window for memory extraction
	var contextWindow strings.Builder
	for _, message := range a.Params.Messages {
		switch message.Role {
		case "user":
			contextWindow.WriteString(fmt.Sprintf("User: %s\n", message.Content))
		case "assistant":
			contextWindow.WriteString(fmt.Sprintf("Assistant: %s\n", message.Content))
		}
	}

	// Extract memories from the conversation
	memExtractSpinner := output.StartSpinner("Processing and storing memories")
	a.extractMemories(ctx, contextWindow.String())
	output.StopSpinner(memExtractSpinner, "New memories stored")

	// Final reporting
	output.Result(fmt.Sprintf("Agent %s completed execution (%d tokens)", a.Name, estimateTokens(fullResponse.String())))

	return fullResponse.String(), nil
}

// Estimate tokens in a string (very rough approximation)
func estimateTokens(text string) int {
	words := strings.Fields(text)
	return len(words) * 4 / 3 // Rough estimate: ~4/3 tokens per word
}

func (a *BaseAgent) createPlan(ctx context.Context, message LLMMessage) (Plan, error) {
	if a.Planner != nil {
		output.Verbose(fmt.Sprintf("Creating plan for: %s", output.Summarize(message.Content, 40)))

		plan, err := a.Planner.CreatePlan(ctx, message.Content)
		if err != nil {
			return Plan{}, fmt.Errorf("failed to create plan: %w", err)
		}

		// Log the plan steps
		output.Verbose(fmt.Sprintf("Plan created with %d steps", len(plan.Steps)))
		for i, step := range plan.Steps {
			output.Debug(fmt.Sprintf("Step %d: %s (Tool: %s)", i+1, step.Description, step.ToolName))
		}

		return plan, nil
	}

	return Plan{}, nil
}

func (a *BaseAgent) injectMemories(ctx context.Context, message LLMMessage) LLMMessage {
	enhancedMessage := message // Create a copy to modify

	if a.Memory != nil {
		// Check if the memory system supports memory preparation
		if memoryEnhancer, ok := a.Memory.(MemoryEnhancer); ok {
			enhancedContext, err := memoryEnhancer.PrepareContext(ctx, a.Name, message.Content)
			if err == nil && enhancedContext != "" {
				output.Verbose(fmt.Sprintf("Enhanced input with memories (%d → %d chars)",
					len(message.Content), len(enhancedContext)))

				enhancedMessage.Content = enhancedContext
				errnie.Info(fmt.Sprintf("Enhanced input with %d characters of memories",
					len(enhancedContext)-len(message.Content)))
			} else if err != nil {
				output.Debug(fmt.Sprintf("Memory enhancement failed: %v", err))
			} else {
				output.Debug("No relevant memories found")
			}
		} else {
			output.Debug("Memory system does not support context enhancement")
		}
	} else {
		output.Debug("No memory system available")
	}

	return enhancedMessage
}

func (a *BaseAgent) extractMemories(ctx context.Context, contextWindow string) {
	if a.Memory != nil {
		if memoryExtractor, ok := a.Memory.(MemoryExtractor); ok {
			output.Verbose("Extracting memories from conversation")

			memories, err := memoryExtractor.ExtractMemories(ctx, a.Name, contextWindow, "conversation")

			if err != nil {
				output.Error("Memory extraction failed", err)
				errnie.Error(err)
			} else if memories != nil {
				output.Result(fmt.Sprintf("Extracted %d memories", len(memories)))
			}
		} else {
			output.Debug("Memory system does not support memory extraction")
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

		output.Action("agent", "execute_tool", toolName)
		toolSpinner := output.StartSpinner(fmt.Sprintf("Executing tool: %s", toolName))

		// Execute the tool
		for _, tool := range a.Tools {
			if tool.Name() == toolName {
				result, err := tool.Execute(ctx, args)

				output.StopSpinner(toolSpinner, "")

				if err != nil {
					output.Error(fmt.Sprintf("Tool %s execution failed", toolName), err)
					return nil, err
				}

				output.Result(fmt.Sprintf("Tool %s executed successfully", toolName))
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
	output.Verbose(fmt.Sprintf("Added tool %s to agent %s", tool.Name(), a.Name))
	return nil
}

// SetMemory sets the memory system for the agent
func (a *BaseAgent) SetMemory(memory Memory) {
	a.Memory = memory
	output.Verbose(fmt.Sprintf("Set memory system for agent %s", a.Name))
}

// SetLLM sets the LLM provider for the agent
func (a *BaseAgent) SetLLM(llm LLMProvider) {
	a.LLM = llm
	output.Verbose(fmt.Sprintf("Set LLM provider %s for agent %s", llm.Name(), a.Name))
}

// SetSystemPrompt sets the system prompt for the agent
func (a *BaseAgent) SetSystemPrompt(prompt string) {
	a.Params.Messages = append(a.Params.Messages, SystemMessage(prompt))
	output.Verbose(fmt.Sprintf("Set system prompt for agent %s (%d chars)", a.Name, len(prompt)))
}

// SetModel sets the model for the agent
func (a *BaseAgent) SetModel(model string) {
	a.Params.Model = model
	output.Verbose(fmt.Sprintf("Set model %s for agent %s", model, a.Name))
}

// SetTemperature sets the temperature for the agent
func (a *BaseAgent) SetTemperature(temperature float64) {
	a.Params.Temperature = temperature
	output.Verbose(fmt.Sprintf("Set temperature %.2f for agent %s", temperature, a.Name))
}

// SetIterationLimit sets the iteration limit for the agent
func (a *BaseAgent) SetIterationLimit(limit int) {
	a.IterationLimit = limit
	output.Verbose(fmt.Sprintf("Set iteration limit %d for agent %s", limit, a.Name))
}

// SetPlanner sets the planner for the agent
func (a *BaseAgent) SetPlanner(planner Planner) {
	a.Planner = planner
	output.Verbose(fmt.Sprintf("Set planner for agent %s", a.Name))
}

// ExecuteWithPlanner executes the agent's task using the planner to guide execution
func (a *BaseAgent) ExecuteWithPlanner(ctx context.Context, message LLMMessage) (string, error) {
	if a.Planner == nil {
		output.Info("No planner set, falling back to standard execution")
		return a.Execute(ctx, message)
	}

	output.Info("Executing with guided planning")

	// First create a plan
	plan, err := a.createPlan(ctx, message)
	if err != nil {
		output.Error("Failed to create plan", err)
		return "", err
	}

	// Use the planner to guide the agent through the plan
	return a.Planner.GuideAgent(ctx, a, plan, message.Content)
}

// GetMessenger returns the agent's messenger
func (a *BaseAgent) GetMessenger() Messenger {
	return a.Messenger
}

// SetMessenger sets the agent's messenger
func (a *BaseAgent) SetMessenger(messenger Messenger) {
	a.Messenger = messenger
	output.Verbose(fmt.Sprintf("Set messenger for agent %s", a.Name))
}

// HandleMessage processes an incoming message
func (a *BaseAgent) HandleMessage(ctx context.Context, message Message) (string, error) {
	// This is a basic implementation - specific agent types can override this
	if a.LLM == nil {
		return "", errors.New("no LLM provider set")
	}

	output.Action("agent", "message", fmt.Sprintf("from %s", message.Sender))

	// For direct messages, we'll just pass the content to the LLM
	if message.IsDirect() {
		promptTemplate := "Message from %s: %s\nPlease respond to this message."
		prompt := fmt.Sprintf(promptTemplate, message.Sender, message.Content)

		// Use the agent's LLM to generate a response
		messageSpinner := output.StartSpinner("Generating response to message")

		res := a.LLM.GenerateResponse(ctx, LLMParams{
			Messages: []LLMMessage{
				{
					Role:    "system",
					Content: prompt,
				},
			},
		})

		if res.Error != nil {
			output.StopSpinner(messageSpinner, "")
			output.Error("Failed to generate response", res.Error)
			return "", res.Error
		}

		output.StopSpinner(messageSpinner, "Response generated")

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
	output.Debug(logMsg)

	return "", nil
}

// RunWorkflow executes a predefined workflow
func (a *BaseAgent) RunWorkflow(ctx context.Context, workflow Workflow, input map[string]interface{}) (map[string]interface{}, error) {
	if workflow == nil {
		return nil, errors.New("workflow is nil")
	}

	output.Action("agent", "workflow", fmt.Sprintf("Agent %s starting workflow execution", a.Name))
	workflowSpinner := output.StartSpinner("Executing workflow")

	// Execute the workflow
	results, err := workflow.Execute(ctx, input)

	if err != nil {
		output.StopSpinner(workflowSpinner, "")
		output.Error("Workflow execution failed", err)
		return nil, err
	}

	output.StopSpinner(workflowSpinner, fmt.Sprintf("Workflow completed with %d results", len(results)))

	return results, nil
}

// formatStreamedContent applies basic formatting to streamed content
func formatStreamedContent(content string) string {
	// For research output, we'll format markdown headers and lists

	// Check for markdown headers
	if strings.HasPrefix(content, "# ") {
		// Main header - use cyan bold
		return color.New(color.FgHiCyan, color.Bold).Sprint(content)
	} else if strings.HasPrefix(content, "## ") {
		// Secondary header - use cyan
		return color.New(color.FgCyan, color.Bold).Sprint(content)
	} else if strings.HasPrefix(content, "### ") {
		// Tertiary header - use blue
		return color.New(color.FgBlue, color.Bold).Sprint(content)
	} else if strings.HasPrefix(content, "- ") || strings.HasPrefix(content, "* ") {
		// List item - use green
		return color.New(color.FgGreen).Sprint(content)
	} else if strings.HasPrefix(content, "> ") {
		// Blockquote - use yellow
		return color.New(color.FgYellow).Sprint(content)
	} else if strings.HasPrefix(content, "```") || strings.HasPrefix(content, "`") {
		// Code - use magenta
		return color.New(color.FgHiMagenta).Sprint(content)
	}

	// Regular text
	return content
}
