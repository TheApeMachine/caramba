package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/theapemachine/caramba/pkg/output"
	"github.com/theapemachine/errnie"
)

// BaseAgent provides a base implementation of the Agent interface.
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

// NewBaseAgent creates a new BaseAgent.
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

// Execute runs the agent with the provided input and returns a response (non-streaming).
func (a *BaseAgent) Execute(ctx context.Context, message LLMMessage) (string, error) {
	// Prepare everything (inject memories, plan, etc.)
	message = a.prepareForExecution(ctx, message)

	// Now run the iteration loop (non-streaming).
	finalResponse, err := a.runIterations(ctx, message, false)
	if err != nil {
		return "", err
	}

	// Extract/store memories from the conversation.
	a.storeMemories(ctx, finalResponse)

	// Final reporting
	output.Result(fmt.Sprintf("Agent %s completed execution (%d tokens)", a.Name, estimateTokens(finalResponse)))
	return finalResponse, nil
}

// StreamExecute runs the agent with the provided input and streams the response in real-time.
func (a *BaseAgent) StreamExecute(ctx context.Context, message LLMMessage) (string, error) {
	// Prepare everything (inject memories, plan, etc.)
	message = a.prepareForExecution(ctx, message)

	// Now run the iteration loop (streaming).
	finalResponse, err := a.runIterations(ctx, message, true)
	if err != nil {
		return "", err
	}

	// Extract/store memories from the conversation.
	a.storeMemories(ctx, finalResponse)

	// Final reporting
	output.Result(fmt.Sprintf("Agent %s completed execution (%d tokens)", a.Name, estimateTokens(finalResponse)))
	return finalResponse, nil
}

// prepareForExecution injects relevant memories, and if a planner is available, creates a plan.
func (a *BaseAgent) prepareForExecution(ctx context.Context, message LLMMessage) LLMMessage {
	if a.LLM == nil {
		output.Error("No LLM provider set", errors.New("missing LLM"))
		return message
	}

	// Log start
	output.Action("agent", "prepare", fmt.Sprintf("%s processing: %s", a.Name, output.Summarize(message.Content, 60)))

	// 1) Inject memories
	memorySpinner := output.StartSpinner("Checking memory for relevant context")
	enhancedMessage := a.injectMemories(ctx, message)
	output.StopSpinner(memorySpinner, "Memory retrieval complete")

	// 2) Create a plan if planner is available
	if a.Planner != nil {
		planSpinner := output.StartSpinner("Planning execution strategy")
		_, err := a.createPlan(ctx, enhancedMessage)
		if err != nil {
			output.StopSpinner(planSpinner, "")
			output.Error("Planning failed", err)
		} else {
			output.StopSpinner(planSpinner, "Execution plan created")
		}
	}

	return enhancedMessage
}

// runIterations handles the iteration loop, either in streaming or non-streaming mode.
func (a *BaseAgent) runIterations(ctx context.Context, initialMsg LLMMessage, streaming bool) (string, error) {
	var response strings.Builder

	iteration := 0
	iterMsg := LLMMessage{
		Role:    initialMsg.Role,
		Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, initialMsg.Content),
	}

	for iteration < a.IterationLimit {
		// Add the current message to the conversation.
		a.Params.Messages = append(a.Params.Messages, iterMsg)

		// Sync tools with LLM params.
		a.toolsMu.RLock()
		a.Params.Tools = a.Tools
		a.toolsMu.RUnlock()

		// Show the spinner if needed.
		iterationStr := fmt.Sprintf("Iteration %d/%d", iteration+1, a.IterationLimit)
		thinkingSpinner := output.StartSpinner(fmt.Sprintf("%s: Agent thinking", iterationStr))

		if !streaming {
			// NON-STREAMING branch:
			res := a.LLM.GenerateResponse(ctx, a.Params)
			if res.Error != nil {
				output.StopSpinner(thinkingSpinner, "")
				output.Error("LLM response generation failed", res.Error)
				return "", res.Error
			}

			// Handle tool calls if any
			if len(res.ToolCalls) > 0 {
				output.StopSpinner(thinkingSpinner, fmt.Sprintf("Agent is using %d tools", len(res.ToolCalls)))
				for _, toolCall := range res.ToolCalls {
					output.Action("agent", "tool_call", toolCall.Name)
				}
			} else {
				output.StopSpinner(thinkingSpinner, "Agent completed thinking")
			}

			// Append response
			response.WriteString(res.Content)
		} else {
			// STREAMING branch:
			output.StopSpinner(thinkingSpinner, "Agent is responding in real-time:")
			output.Info("Streaming response begins:")
			fmt.Println(strings.Repeat("-", 40))

			// We'll accumulate the chunks in a strings.Builder.
			var streamedResponse strings.Builder
			toolCallsFound := false
			var toolCalls []ToolCall

			streamChan := a.LLM.StreamResponse(ctx, a.Params)
			for chunk := range streamChan {
				if chunk.Error != nil {
					output.Error("Streaming failed", chunk.Error)
					return response.String(), chunk.Error
				}
				if len(chunk.ToolCalls) > 0 {
					toolCallsFound = true
					toolCalls = chunk.ToolCalls

					// Log the tool calls
					for _, toolCall := range chunk.ToolCalls {
						argsSummary := summarizeToolCallArgs(toolCall)
						output.Action("agent", "tool_call", fmt.Sprintf("%s: %s", toolCall.Name, argsSummary))
					}
				} else if chunk.Content != "" {
					content := chunk.Content
					if isLikelyJSON(content) {
						// Attempt to extract "content" field if it's JSON
						content = maybeExtractContentField(content)
					}
					if content != "" {
						// Format and print for real-time streaming
						formatted := formatStreamedContent(content)
						fmt.Print(formatted)
						streamedResponse.WriteString(content)
					}
				}
			}
			fmt.Println()
			fmt.Println(strings.Repeat("-", 40))
			output.Info("Streaming response complete")

			// If tools were discovered in this streaming chunk, run them.
			if toolCallsFound {
				output.Info(fmt.Sprintf("Processing %d tool calls", len(toolCalls)))
				for _, tc := range toolCalls {
					fmt.Println()
					fmt.Println(strings.Repeat("=", 40))
					output.Title(fmt.Sprintf("TOOL CALL: %s", strings.ToUpper(tc.Name)))

					prettyArgs, _ := json.MarshalIndent(tc.Args, "", "  ")
					fmt.Println("Arguments:")
					fmt.Println(string(prettyArgs))

					output.Action("agent", "execute_tool", tc.Name)
					toolSpinner := output.StartSpinner(fmt.Sprintf("Executing tool: %s", tc.Name))

					toolFound := false
					for _, tool := range a.Tools {
						if tool.Name() == tc.Name {
							toolFound = true
							result, err := tool.Execute(ctx, tc.Args)
							if err != nil {
								output.StopSpinner(toolSpinner, "")
								output.Error(fmt.Sprintf("Tool %s execution failed", tc.Name), err)
								continue
							}
							output.StopSpinner(toolSpinner, fmt.Sprintf("Tool %s executed successfully", tc.Name))

							// Pretty print the result
							toolResultStr := formatToolResult(tc.Name, result)
							fmt.Println(toolResultStr)
							fmt.Println(strings.Repeat("=", 40))
							fmt.Println()

							// Add to the streamed response
							streamedResponse.WriteString(fmt.Sprintf("\n\nTool Result (%s):\n%v\n\n", tc.Name, toolResultStr))
						}
					}
					if !toolFound {
						output.StopSpinner(toolSpinner, "")
						output.Error(fmt.Sprintf("Tool %s not found", tc.Name), nil)
					}
				}
			}
			// Accumulate everything into final response
			response.WriteString(streamedResponse.String())
		}

		// Next iteration
		iteration++
		iterMsg = LLMMessage{
			Role:    "assistant",
			Content: fmt.Sprintf("Iteration %d\n\n%s", iteration, response.String()),
		}
	}

	return response.String(), nil
}

// storeMemories collects conversation from a.Params.Messages, then calls memory extraction (if available).
func (a *BaseAgent) storeMemories(ctx context.Context, finalResponse string) {
	var contextWindow strings.Builder
	for _, msg := range a.Params.Messages {
		switch msg.Role {
		case "user":
			contextWindow.WriteString(fmt.Sprintf("User: %s\n", msg.Content))
		case "assistant":
			contextWindow.WriteString(fmt.Sprintf("Assistant: %s\n", msg.Content))
		}
	}

	memExtractSpinner := output.StartSpinner("Processing and storing memories")
	a.extractMemories(ctx, contextWindow.String())
	output.StopSpinner(memExtractSpinner, "New memories stored")
}

// createPlan attempts to create a plan via the Planner, if present.
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

// GetToolResults processes the results of tool calls for the iterator (placeholder).
func (a *BaseAgent) GetToolResults(ctx context.Context, response string) (map[string]interface{}, error) {
	// This is a simplified placeholder in the original code
	toolResults := make(map[string]interface{})
	toolCalls := extractToolCalls(response)
	for _, tc := range toolCalls {
		output.Action("agent", "execute_tool", tc.Name)
		toolSpinner := output.StartSpinner(fmt.Sprintf("Executing tool: %s", tc.Name))

		for _, tool := range a.Tools {
			if tool.Name() == tc.Name {
				result, err := tool.Execute(ctx, tc.Args)
				output.StopSpinner(toolSpinner, "")
				if err != nil {
					output.Error(fmt.Sprintf("Tool %s execution failed", tc.Name), err)
					return nil, err
				}
				output.Result(fmt.Sprintf("Tool %s executed successfully", tc.Name))
				toolResults[tc.Name] = result
			}
		}
	}
	return toolResults, nil
}

// ToolCall is a parsed tool call from the agent's response.
type ToolCall struct {
	Name string
	Args map[string]interface{}
}

// extractToolCalls tries to parse a response for tool calls (placeholder).
func extractToolCalls(response string) []ToolCall {
	var toolCalls []ToolCall
	if err := json.Unmarshal([]byte(response), &toolCalls); err == nil && len(toolCalls) > 0 {
		return toolCalls
	}
	// Try single tool call
	var singleToolCall ToolCall
	if err := json.Unmarshal([]byte(response), &singleToolCall); err == nil && singleToolCall.Name != "" {
		return []ToolCall{singleToolCall}
	}
	// Fallback: no recognized calls
	return []ToolCall{}
}

// AddTool adds a new tool to the agent.
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

// SetMemory sets the memory system for the agent.
func (a *BaseAgent) SetMemory(memory Memory) {
	a.Memory = memory
	output.Verbose(fmt.Sprintf("Set memory system for agent %s", a.Name))
}

// SetLLM sets the LLM provider for the agent.
func (a *BaseAgent) SetLLM(llm LLMProvider) {
	a.LLM = llm
	output.Verbose(fmt.Sprintf("Set LLM provider %s for agent %s", llm.Name(), a.Name))
}

// SetSystemPrompt sets the system prompt for the agent.
func (a *BaseAgent) SetSystemPrompt(prompt string) {
	a.Params.Messages = append(a.Params.Messages, SystemMessage(prompt))
	output.Verbose(fmt.Sprintf("Set system prompt for agent %s (%d chars)", a.Name, len(prompt)))
}

// SetModel sets the model for the agent.
func (a *BaseAgent) SetModel(model string) {
	a.Params.Model = model
	output.Verbose(fmt.Sprintf("Set model %s for agent %s", model, a.Name))
}

// SetTemperature sets the temperature for the agent.
func (a *BaseAgent) SetTemperature(temperature float64) {
	a.Params.Temperature = temperature
	output.Verbose(fmt.Sprintf("Set temperature %.2f for agent %s", temperature, a.Name))
}

// SetIterationLimit sets the iteration limit for the agent.
func (a *BaseAgent) SetIterationLimit(limit int) {
	a.IterationLimit = limit
	output.Verbose(fmt.Sprintf("Set iteration limit %d for agent %s", limit, a.Name))
}

// SetPlanner sets the planner for the agent.
func (a *BaseAgent) SetPlanner(planner Planner) {
	a.Planner = planner
	output.Verbose(fmt.Sprintf("Set planner for agent %s", a.Name))
}

// ExecuteWithPlanner runs the agent's task using the planner to guide execution.
func (a *BaseAgent) ExecuteWithPlanner(ctx context.Context, message LLMMessage) (string, error) {
	if a.Planner == nil {
		output.Info("No planner set, falling back to standard execution")
		return a.Execute(ctx, message)
	}

	output.Info("Executing with guided planning")
	plan, err := a.createPlan(ctx, message)
	if err != nil {
		output.Error("Failed to create plan", err)
		return "", err
	}
	return a.Planner.GuideAgent(ctx, a, plan, message.Content)
}

// GetMessenger returns the agent's messenger.
func (a *BaseAgent) GetMessenger() Messenger {
	return a.Messenger
}

// SetMessenger sets the agent's messenger.
func (a *BaseAgent) SetMessenger(messenger Messenger) {
	a.Messenger = messenger
	output.Verbose(fmt.Sprintf("Set messenger for agent %s", a.Name))
}

// HandleMessage processes an incoming Message (simplistic in this example).
func (a *BaseAgent) HandleMessage(ctx context.Context, message Message) (string, error) {
	if a.LLM == nil {
		return "", errors.New("no LLM provider set")
	}
	output.Action("agent", "message", fmt.Sprintf("from %s", message.Sender))

	// If direct message, generate a quick response.
	if message.IsDirect() {
		promptTemplate := "Message from %s: %s\nPlease respond to this message."
		prompt := fmt.Sprintf(promptTemplate, message.Sender, message.Content)
		messageSpinner := output.StartSpinner("Generating response to message")

		res := a.LLM.GenerateResponse(ctx, LLMParams{
			Messages: []LLMMessage{
				{Role: "system", Content: prompt},
			},
		})
		if res.Error != nil {
			output.StopSpinner(messageSpinner, "")
			output.Error("Failed to generate response", res.Error)
			return "", res.Error
		}

		output.StopSpinner(messageSpinner, "Response generated")
		// Optionally respond back
		if a.Messenger != nil {
			_, _ = a.Messenger.SendDirect(ctx, message.Sender, res.Content, MessageTypeText, nil)
		}
		return res.Content, nil
	}

	// Otherwise, just log
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

// RunWorkflow executes a predefined workflow.
func (a *BaseAgent) RunWorkflow(ctx context.Context, workflow Workflow, input map[string]interface{}) (map[string]interface{}, error) {
	if workflow == nil {
		return nil, errors.New("workflow is nil")
	}

	output.Action("agent", "workflow", fmt.Sprintf("Agent %s starting workflow execution", a.Name))
	workflowSpinner := output.StartSpinner("Executing workflow")

	results, err := workflow.Execute(ctx, input)
	if err != nil {
		output.StopSpinner(workflowSpinner, "")
		output.Error("Workflow execution failed", err)
		return nil, err
	}

	output.StopSpinner(workflowSpinner, fmt.Sprintf("Workflow completed with %d results", len(results)))
	return results, nil
}
