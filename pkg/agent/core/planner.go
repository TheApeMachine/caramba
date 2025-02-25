package core

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// SimplePlanner implements the Planner interface
type SimplePlanner struct {
	llm      LLMProvider
	tools    map[string]Tool
	maxSteps int
}

// NewSimplePlanner creates a new SimplePlanner
func NewSimplePlanner(llm LLMProvider, tools map[string]Tool) *SimplePlanner {
	return &SimplePlanner{
		llm:      llm,
		tools:    tools,
		maxSteps: 10,
	}
}

// CreatePlan creates a plan based on the input
func (p *SimplePlanner) CreatePlan(ctx context.Context, input string) (Plan, error) {
	if p.llm == nil {
		return Plan{}, fmt.Errorf("no LLM provider set")
	}
	
	// Build a prompt for the LLM to create a plan
	toolDescriptions := p.buildToolDescriptions()
	
	prompt := fmt.Sprintf(`You are an AI assistant that creates plans to solve tasks.
Given the user's request, create a step-by-step plan using available tools.

Available tools:
%s

User request: %s

Respond with a JSON array of steps, where each step is an object with:
- "description": A description of the step
- "tool": The name of the tool to use, or null if no tool is needed
- "arguments": A JSON object of arguments for the tool

Example response:
[
  {
    "description": "Calculate the sum of 2+2",
    "tool": "calculator",
    "arguments": {
      "expression": "2+2"
    }
  }
]

Your plan:`, toolDescriptions, input)
	
	// Ask the LLM to create a plan
	options := LLMOptions{
		MaxTokens:    1024,
		Temperature:  0.7,
		SystemPrompt: "You are a helpful AI assistant that creates plans to solve tasks.",
	}
	
	response, err := p.llm.GenerateResponse(ctx, prompt, options)
	if err != nil {
		return Plan{}, fmt.Errorf("failed to generate plan: %w", err)
	}
	
	// Extract the JSON part of the response
	jsonPlan := extractJSON(response)
	if jsonPlan == "" {
		return Plan{}, fmt.Errorf("failed to extract JSON plan from response")
	}
	
	// Parse the JSON plan
	var planSteps []struct {
		Description string                 `json:"description"`
		Tool        string                 `json:"tool"`
		Arguments   map[string]interface{} `json:"arguments"`
	}
	
	if err := json.Unmarshal([]byte(jsonPlan), &planSteps); err != nil {
		return Plan{}, fmt.Errorf("failed to parse plan: %w", err)
	}
	
	// Convert to the Plan format
	plan := Plan{
		Steps: make([]PlanStep, len(planSteps)),
	}
	
	for i, step := range planSteps {
		plan.Steps[i] = PlanStep{
			Description: step.Description,
			ToolName:    step.Tool,
			Arguments:   step.Arguments,
		}
	}
	
	return plan, nil
}

// ExecutePlan executes a plan and returns the result
func (p *SimplePlanner) ExecutePlan(ctx context.Context, plan Plan) (string, error) {
	if len(plan.Steps) == 0 {
		return "No steps to execute", nil
	}
	
	// Execute each step
	results := make([]string, 0, len(plan.Steps))
	
	for i, step := range plan.Steps {
		if i >= p.maxSteps {
			results = append(results, "Plan execution stopped - reached maximum number of steps")
			break
		}
		
		result, err := p.executeStep(ctx, step)
		if err != nil {
			results = append(results, fmt.Sprintf("Step %d failed: %s", i+1, err.Error()))
			break
		}
		
		results = append(results, fmt.Sprintf("Step %d: %s", i+1, result))
	}
	
	// Join the results
	return strings.Join(results, "\n"), nil
}

// executeStep executes a single step
func (p *SimplePlanner) executeStep(ctx context.Context, step PlanStep) (string, error) {
	// If no tool is specified, just return the description
	if step.ToolName == "" {
		return step.Description, nil
	}
	
	// Find the tool
	tool, exists := p.tools[step.ToolName]
	if !exists {
		return "", fmt.Errorf("tool not found: %s", step.ToolName)
	}
	
	// Execute the tool
	result, err := tool.Execute(ctx, step.Arguments)
	if err != nil {
		return "", fmt.Errorf("failed to execute tool %s: %w", step.ToolName, err)
	}
	
	// Convert the result to a string
	var resultStr string
	switch r := result.(type) {
	case string:
		resultStr = r
	case []byte:
		resultStr = string(r)
	default:
		// Try to marshal to JSON
		jsonBytes, err := json.Marshal(result)
		if err != nil {
			resultStr = fmt.Sprintf("%v", result)
		} else {
			resultStr = string(jsonBytes)
		}
	}
	
	return resultStr, nil
}

// buildToolDescriptions builds a description of available tools
func (p *SimplePlanner) buildToolDescriptions() string {
	var builder strings.Builder
	
	for name, tool := range p.tools {
		builder.WriteString(fmt.Sprintf("- %s: %s\n", name, tool.Description()))
		
		// Add schema information if available
		schema := tool.Schema()
		if schema != nil {
			if properties, ok := schema["properties"].(map[string]interface{}); ok {
				builder.WriteString("  Arguments:\n")
				for argName, argInfo := range properties {
					if argInfoMap, ok := argInfo.(map[string]interface{}); ok {
						description := argInfoMap["description"]
						if description != nil {
							builder.WriteString(fmt.Sprintf("    - %s: %s\n", argName, description))
						} else {
							builder.WriteString(fmt.Sprintf("    - %s\n", argName))
						}
					}
				}
			}
		}
		
		builder.WriteString("\n")
	}
	
	return builder.String()
}

// extractJSON extracts a JSON array or object from a string
func extractJSON(s string) string {
	// Look for JSON array
	arrayStart := strings.Index(s, "[")
	arrayEnd := strings.LastIndex(s, "]")
	if arrayStart != -1 && arrayEnd != -1 && arrayEnd > arrayStart {
		return s[arrayStart : arrayEnd+1]
	}
	
	// Look for JSON object
	objectStart := strings.Index(s, "{")
	objectEnd := strings.LastIndex(s, "}")
	if objectStart != -1 && objectEnd != -1 && objectEnd > objectStart {
		return s[objectStart : objectEnd+1]
	}
	
	return ""
}
