package core

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/pkg/output"
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

	res := p.llm.GenerateResponse(ctx, LLMParams{
		Messages: []LLMMessage{
			{
				Role:    "system",
				Content: viper.GetViper().GetString("templates.planner"),
			},
			{
				Role:    "user",
				Content: fmt.Sprintf("Given the user's request, create a step-by-step plan using available tools. User request: %s", input),
			},
		},
	})
	if res.Error != nil {
		return Plan{}, fmt.Errorf("failed to generate plan: %w", res.Error)
	}

	// Extract the JSON part of the response
	jsonPlan := extractJSON(res.Content)
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

// GuideAgent executes a plan with an agent as the executor, where the planner
// guides the agent through each step of the plan
func (p *SimplePlanner) GuideAgent(ctx context.Context, agent Agent, plan Plan, query string) (string, error) {
	if len(plan.Steps) == 0 {
		return "No steps to execute", nil
	}

	// Build a complete result from all steps
	var compiledResults strings.Builder

	// Execute each plan step by guiding the agent
	for i, step := range plan.Steps {
		if i >= p.maxSteps {
			output.Info("Plan execution stopped - reached maximum number of steps")
			break
		}

		output.Info(fmt.Sprintf("Executing step %d: %s", i+1, step.Description))
		stepSpinner := output.StartSpinner(fmt.Sprintf("Working on step %d/%d", i+1, len(plan.Steps)))

		// Create a message that includes the plan context for the agent
		stepContext := fmt.Sprintf(
			"You are working on a plan for: %s\n\n"+
				"The current step (%d/%d) is: %s\n\n"+
				"If this step requires using a tool, use the '%s' tool with appropriate arguments.\n\n"+
				"Focus only on completing this specific step and report your findings.",
			query, i+1, len(plan.Steps), step.Description, step.ToolName)

		// Inject the step context into the agent's execution
		stepResult, err := agent.Execute(ctx, LLMMessage{
			Role:    "user",
			Content: stepContext,
		})

		if err != nil {
			output.StopSpinner(stepSpinner, fmt.Sprintf("Step %d failed", i+1))
			output.Error(fmt.Sprintf("Failed to execute step %d", i+1), err)
			continue
		}

		output.StopSpinner(stepSpinner, fmt.Sprintf("Step %d completed", i+1))

		// Validate step completion
		validationResult, feedback := p.validateStepCompletion(ctx, step, stepResult)

		if !validationResult {
			output.Info(fmt.Sprintf("Step %d may not be fully complete: %s", i+1, feedback))

			// Attempt correction if needed
			if feedback != "" {
				output.Info("Attempting to correct step execution")
				correctionSpinner := output.StartSpinner("Correcting step execution")

				correctedContext := fmt.Sprintf(
					"%s\n\nYour previous attempt had these issues: %s\n\nPlease try again and address these issues.",
					stepContext, feedback)

				correctedResult, err := agent.Execute(ctx, LLMMessage{
					Role:    "user",
					Content: correctedContext,
				})

				if err == nil {
					stepResult = correctedResult
					output.StopSpinner(correctionSpinner, "Step correction successful")
				} else {
					output.StopSpinner(correctionSpinner, "Step correction failed")
					output.Error("Failed to correct step", err)
				}
			}
		}

		// Add this step's results to the overall compilation
		compiledResults.WriteString(fmt.Sprintf("\n\n## Step %d: %s\n\n", i+1, step.Description))
		compiledResults.WriteString(stepResult)
	}

	// Synthesize the results if we have a valid LLM
	if p.llm != nil {
		synthesisSpinner := output.StartSpinner("Creating final synthesized report")

		synthesisPrompt := fmt.Sprintf(
			"You have completed a plan for: %s\n\n"+
				"Here are the results from each step of your plan:\n\n%s\n\n"+
				"Please synthesize these findings into a comprehensive, well-structured report in markdown format. "+
				"Include key insights, analysis, and conclusions.",
			query, compiledResults.String())

		synthesisResponse := p.llm.GenerateResponse(ctx, LLMParams{
			Messages: []LLMMessage{
				{
					Role:    "system",
					Content: "You are an expert at synthesizing information into clear, comprehensive reports.",
				},
				{
					Role:    "user",
					Content: synthesisPrompt,
				},
			},
		})

		if synthesisResponse.Error == nil {
			output.StopSpinner(synthesisSpinner, "Final report synthesized")
			return synthesisResponse.Content, nil
		} else {
			output.StopSpinner(synthesisSpinner, "Failed to synthesize final report")
			output.Error("Synthesis failed", synthesisResponse.Error)
		}
	}

	// If synthesis failed or no LLM is available, return the raw compiled results
	return compiledResults.String(), nil
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

// validateStepCompletion checks if a step was completed successfully
// Returns true if valid, and any feedback for improvement
func (p *SimplePlanner) validateStepCompletion(ctx context.Context, step PlanStep, result string) (bool, string) {
	if p.llm == nil {
		// If no LLM, assume it's valid
		return true, ""
	}

	validationPrompt := fmt.Sprintf(
		"You are validating whether the following step was completed correctly:\n\n"+
			"Step description: %s\n\n"+
			"Expected tool: %s\n\n"+
			"Result produced: %s\n\n"+
			"Please determine if the step was completed correctly and provide feedback in JSON format: "+
			"{\n  \"is_complete\": true/false,\n  \"feedback\": \"reasons why the step is incomplete or suggestions for improvement\"\n}",
		step.Description, step.ToolName, result)

	validationResponse := p.llm.GenerateResponse(ctx, LLMParams{
		Messages: []LLMMessage{
			{
				Role:    "user",
				Content: validationPrompt,
			},
		},
	})

	if validationResponse.Error != nil {
		output.Info("Step validation failed, assuming step is complete")
		return true, ""
	}

	// Extract JSON response
	jsonStr := extractJSON(validationResponse.Content)
	if jsonStr == "" {
		output.Info("Could not parse validation response, assuming step is complete")
		return true, ""
	}

	// Parse the validation result
	type ValidationResult struct {
		IsComplete bool   `json:"is_complete"`
		Feedback   string `json:"feedback"`
	}

	var validationResult ValidationResult
	err := json.Unmarshal([]byte(jsonStr), &validationResult)
	if err != nil {
		output.Info("Failed to parse validation JSON, assuming step is complete")
		return true, ""
	}

	return validationResult.IsComplete, validationResult.Feedback
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
