package core

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

// ToolManager handles management and execution of agent tools.
type ToolManager struct {
	logger            *output.Logger
	hub               *hub.Queue
	tools             []Tool
	toolsMu           sync.RWMutex
	responseProcessor *ResponseProcessor
}

// NewToolManager creates a new ToolManager.
func NewToolManager() *ToolManager {
	return &ToolManager{
		logger:            output.NewLogger(),
		hub:               hub.NewQueue(),
		tools:             make([]Tool, 0),
		responseProcessor: NewResponseProcessor(),
	}
}

// AddTool adds a new tool to the manager.
func (tm *ToolManager) AddTool(tool Tool) error {
	tm.logger.Log(fmt.Sprintf("Adding tool %s", tool.Name()))

	tm.toolsMu.Lock()
	defer tm.toolsMu.Unlock()

	for _, existingTool := range tm.tools {
		if existingTool.Name() == tool.Name() {
			return fmt.Errorf("tool %s already exists", tool.Name())
		}
	}
	tm.tools = append(tm.tools, tool)
	output.Verbose(fmt.Sprintf("Added tool %s", tool.Name()))
	return nil
}

// GetTools returns a copy of the tools slice.
func (tm *ToolManager) GetTools() []Tool {
	tm.toolsMu.RLock()
	defer tm.toolsMu.RUnlock()

	// Return a copy to prevent race conditions
	toolsCopy := make([]Tool, len(tm.tools))
	copy(toolsCopy, tm.tools)
	return toolsCopy
}

// logError logs an error to the log file instead of trying to send it through the hub
func (tm *ToolManager) logError(context string, err error) {
	tm.logger.Log(fmt.Sprintf("[ERROR] %s: %v", context, err))
}

// LogToolCalls logs tool calls to the hub and console
func (tm *ToolManager) LogToolCalls(toolCalls []ToolCall, ctx ...context.Context) {
	q := hub.NewQueue()

	for _, toolCall := range toolCalls {
		argsSummary := tm.responseProcessor.SummarizeToolCallArgs(toolCall)

		// Send via hub if available
		q.Add(hub.NewEvent(
			"tool_manager",
			"actions",
			"assistant",
			hub.EventTypeToolCall,
			fmt.Sprintf("%s: %s", toolCall.Name, argsSummary),
			map[string]string{},
		))
	}
}

// ExecuteToolCalls executes a list of tool calls and returns a formatted string of results.
func (tm *ToolManager) ExecuteToolCalls(ctx context.Context, toolCalls []ToolCall) string {
	var toolResults strings.Builder

	q := hub.NewQueue()
	q.Add(hub.NewEvent(
		"tool_manager",
		"ui",
		"assistant",
		hub.EventTypeStatus,
		fmt.Sprintf("Executing %d tool call(s)", len(toolCalls)),
		map[string]string{},
	))

	for _, tc := range toolCalls {
		// Send tool call event to hub if available
		q.Add(hub.NewEvent(
			"tool_manager",
			"actions",
			"assistant",
			hub.EventTypeToolCall,
			tc.Name,
			map[string]string{},
		))

		result := tm.ExecuteSingleToolCall(ctx, tc)
		if result != "" {
			toolResults.WriteString(fmt.Sprintf("\n\nTool Result (%s):\n%v\n\n", tc.Name, result))

			// Also send the result to the hub if available
			q.Add(hub.NewEvent(
				"tool_manager",
				"ui",
				"assistant",
				hub.EventTypeMessage,
				fmt.Sprintf("Tool Result (%s):\n%v", tc.Name, result),
				map[string]string{},
			))
		}
	}

	return toolResults.String()
}

// ExecuteToolCallsToMap executes tool calls and returns results in a map.
func (tm *ToolManager) ExecuteToolCallsToMap(ctx context.Context, toolCalls []ToolCall) (map[string]interface{}, error) {
	toolResults := make(map[string]interface{})

	// Get hub from context if available
	q := hub.NewQueue()

	// Send info about processing tool calls
	q.Add(hub.NewEvent(
		"tool_manager",
		"ui",
		"assistant",
		hub.EventTypeStatus,
		fmt.Sprintf("Processing %d tool calls", len(toolCalls)),
		map[string]string{},
	))

	for _, tc := range toolCalls {
		// Notify about tool execution
		q.Add(hub.NewEvent(
			"tool_manager",
			"actions",
			"assistant",
			hub.EventTypeToolCall,
			tc.Name,
			map[string]string{},
		))

		// Process arguments - handle potential nested JSON in "args" field
		processedArgs := tc.Args
		if argsStr, ok := tc.Args["args"].(string); ok && argsStr != "" {
			// If we have a string "args" field, try to parse it as JSON
			var nestedArgs map[string]interface{}
			if err := json.Unmarshal([]byte(argsStr), &nestedArgs); err == nil {
				// Successfully parsed nested JSON, use these args instead
				processedArgs = nestedArgs
			}
		}

		toolFound := false
		tm.toolsMu.RLock()
		for _, tool := range tm.tools {
			if tool.Name() == tc.Name {
				toolFound = true
				tm.toolsMu.RUnlock()

				result, err := tool.Execute(ctx, processedArgs)
				if err != nil {
					q.Add(hub.NewEvent(
						"tool_manager",
						"ui",
						"assistant",
						hub.EventTypeError,
						fmt.Sprintf("Tool %s execution failed: %v", tc.Name, err),
						map[string]string{},
					))
					tm.logError("Failed to log tool execution error", err)
				}

				q.Add(hub.NewEvent(
					"tool_manager",
					"ui",
					"assistant",
					hub.EventTypeStatus,
					fmt.Sprintf("Tool %s executed successfully", tc.Name),
					map[string]string{},
				))
				tm.logError("Failed to log tool execution status", err)

				toolResults[tc.Name] = result
				goto nextTool // Break out of the loop
			}
		}
		tm.toolsMu.RUnlock()

		if !toolFound {
			q.Add(hub.NewEvent(
				"tool_manager",
				"ui",
				"assistant",
				hub.EventTypeError,
				fmt.Sprintf("Tool %s not found", tc.Name),
				map[string]string{},
			))
		}
	nextTool:
	}

	return toolResults, nil
}

// ExecuteSingleToolCall executes a single tool call and returns the result.
func (tm *ToolManager) ExecuteSingleToolCall(ctx context.Context, tc ToolCall) string {
	// Get hub from context if available
	q := hub.NewQueue()

	q.Add(hub.NewEvent(
		"tool_manager",
		"actions",
		"assistant",
		hub.EventTypeToolCall,
		tc.Name,
		map[string]string{},
	))

	// Process arguments - handle potential nested JSON in "args" field
	processedArgs := tc.Args
	if argsStr, ok := tc.Args["args"].(string); ok && argsStr != "" {
		// If we have a string "args" field, try to parse it as JSON
		var nestedArgs map[string]interface{}
		if err := json.Unmarshal([]byte(argsStr), &nestedArgs); err == nil {
			// Successfully parsed nested JSON, use these args instead
			processedArgs = nestedArgs
		}
	}

	// Look for the tool
	toolFound := false
	var toolResultStr string

	tm.toolsMu.RLock()
	for _, tool := range tm.tools {
		if tool.Name() == tc.Name {
			toolFound = true
			tm.toolsMu.RUnlock()

			result, err := tool.Execute(ctx, processedArgs)
			if err != nil {
				q.Add(hub.NewEvent(
					"tool_manager",
					"ui",
					"assistant",
					hub.EventTypeError,
					fmt.Sprintf("Tool %s execution failed: %v", tc.Name, err),
					map[string]string{},
				))
				tm.logError("Failed to log tool execution error", err)

				return ""
			}

			q.Add(hub.NewEvent(
				"tool_manager",
				"ui",
				"assistant",
				hub.EventTypeStatus,
				fmt.Sprintf("Tool %s executed successfully", tc.Name),
				map[string]string{},
			))
			tm.logError("Failed to log tool execution status", err)

			return tm.responseProcessor.FormatToolResult(tc.Name, result)
		}
	}

	tm.toolsMu.RUnlock()

	if !toolFound {
		errorMsg := fmt.Sprintf("Tool %s not found", tc.Name)

		q.Add(hub.NewEvent(
			"tool_manager",
			"ui",
			"assistant",
			hub.EventTypeError,
			errorMsg,
			map[string]string{},
		))

		return ""
	}

	return toolResultStr
}
