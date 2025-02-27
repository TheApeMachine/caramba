package core

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/theapemachine/caramba/pkg/output"
)

// ToolManager handles management and execution of agent tools.
type ToolManager struct {
	tools             []Tool
	toolsMu           sync.RWMutex
	responseProcessor *ResponseProcessor
}

// NewToolManager creates a new ToolManager.
func NewToolManager() *ToolManager {
	return &ToolManager{
		tools:             make([]Tool, 0),
		responseProcessor: NewResponseProcessor(),
	}
}

// AddTool adds a new tool to the manager.
func (tm *ToolManager) AddTool(tool Tool) error {
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

// LogToolCalls logs information about tool calls.
func (tm *ToolManager) LogToolCalls(toolCalls []ToolCall) {
	for _, toolCall := range toolCalls {
		argsSummary := tm.responseProcessor.SummarizeToolCallArgs(toolCall)
		output.Action("agent", "tool_call", fmt.Sprintf("%s: %s", toolCall.Name, argsSummary))
	}
}

// ExecuteToolCalls executes a list of tool calls and returns a formatted string of results.
func (tm *ToolManager) ExecuteToolCalls(ctx context.Context, toolCalls []ToolCall) string {
	var toolResults strings.Builder

	output.Info(fmt.Sprintf("Processing %d tool calls", len(toolCalls)))

	for _, tc := range toolCalls {
		fmt.Println()
		fmt.Println(strings.Repeat("=", 40))
		output.Title(fmt.Sprintf("TOOL CALL: %s", strings.ToUpper(tc.Name)))

		prettyArgs, _ := json.MarshalIndent(tc.Args, "", "  ")
		fmt.Println("Arguments:")
		fmt.Println(string(prettyArgs))

		result := tm.ExecuteSingleToolCall(ctx, tc)
		if result != "" {
			toolResults.WriteString(fmt.Sprintf("\n\nTool Result (%s):\n%v\n\n", tc.Name, result))
		}
	}

	return toolResults.String()
}

// ExecuteToolCallsToMap executes tool calls and returns results in a map.
func (tm *ToolManager) ExecuteToolCallsToMap(ctx context.Context, toolCalls []ToolCall) (map[string]interface{}, error) {
	toolResults := make(map[string]interface{})

	output.Info(fmt.Sprintf("Processing %d tool calls", len(toolCalls)))

	for _, tc := range toolCalls {
		output.Action("agent", "execute_tool", tc.Name)
		toolSpinner := output.StartSpinner(fmt.Sprintf("Executing tool: %s", tc.Name))

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
					output.StopSpinner(toolSpinner, "")
					output.Error(fmt.Sprintf("Tool %s execution failed", tc.Name), err)
					return nil, err
				}
				output.StopSpinner(toolSpinner, fmt.Sprintf("Tool %s executed successfully", tc.Name))
				toolResults[tc.Name] = result
				goto nextTool // Break out of the loop
			}
		}
		tm.toolsMu.RUnlock()

		if !toolFound {
			output.StopSpinner(toolSpinner, "")
			output.Error(fmt.Sprintf("Tool %s not found", tc.Name), nil)
		}
	nextTool:
	}

	return toolResults, nil
}

// ExecuteSingleToolCall executes a single tool call and returns the result.
func (tm *ToolManager) ExecuteSingleToolCall(ctx context.Context, tc ToolCall) string {
	output.Action("agent", "execute_tool", tc.Name)
	toolSpinner := output.StartSpinner(fmt.Sprintf("Executing tool: %s", tc.Name))

	// Look for the tool
	toolFound := false
	var toolResultStr string

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

	tm.toolsMu.RLock()
	for _, tool := range tm.tools {
		if tool.Name() == tc.Name {
			toolFound = true
			tm.toolsMu.RUnlock()

			result, err := tool.Execute(ctx, processedArgs)
			if err != nil {
				output.StopSpinner(toolSpinner, "")
				output.Error(fmt.Sprintf("Tool %s execution failed", tc.Name), err)
				return ""
			}
			output.StopSpinner(toolSpinner, fmt.Sprintf("Tool %s executed successfully", tc.Name))

			// Pretty print the result
			toolResultStr = tm.responseProcessor.FormatToolResult(tc.Name, result)
			fmt.Println(toolResultStr)
			fmt.Println(strings.Repeat("=", 40))
			fmt.Println()
			return toolResultStr
		}
	}
	tm.toolsMu.RUnlock()

	if !toolFound {
		output.StopSpinner(toolSpinner, "")
		output.Error(fmt.Sprintf("Tool %s not found", tc.Name), nil)
	}

	return ""
}
