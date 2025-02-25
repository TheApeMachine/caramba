package workflow

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/agent/core"
)

// Step represents a workflow step
type Step struct {
	Name      string
	Tool      core.Tool
	Args      map[string]interface{}
	Condition string // Expression to evaluate for conditional execution
}

// BasicWorkflow implements the Workflow interface
type BasicWorkflow struct {
	steps        []Step
	errorHandler func(error) error
}

// NewWorkflow creates a new workflow
func NewWorkflow() *BasicWorkflow {
	return &BasicWorkflow{
		steps: make([]Step, 0),
	}
}

// AddStep adds a step to the workflow
func (w *BasicWorkflow) AddStep(name string, tool core.Tool, args map[string]interface{}) core.Workflow {
	w.steps = append(w.steps, Step{
		Name: name,
		Tool: tool,
		Args: args,
	})
	return w
}

// AddConditionalStep adds a conditional step to the workflow
func (w *BasicWorkflow) AddConditionalStep(name string, condition string, tool core.Tool, args map[string]interface{}) core.Workflow {
	w.steps = append(w.steps, Step{
		Name:      name,
		Tool:      tool,
		Args:      args,
		Condition: condition,
	})
	return w
}

// SetErrorHandler sets a handler for errors
func (w *BasicWorkflow) SetErrorHandler(handler func(error) error) core.Workflow {
	w.errorHandler = handler
	return w
}

// Execute executes the workflow with the given input
func (w *BasicWorkflow) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	results := make(map[string]interface{})

	// Add input to results
	for k, v := range input {
		results[k] = v
	}

	for _, step := range w.steps {
		// Check if the step should be executed
		if step.Condition != "" {
			shouldExecute, err := w.evaluateCondition(step.Condition, results)
			if err != nil {
				return results, fmt.Errorf("failed to evaluate condition for step %s: %w", step.Name, err)
			}

			if !shouldExecute {
				// Skip this step
				continue
			}
		}

		// Process argument templates
		processedArgs, err := w.processArgs(step.Args, results)
		if err != nil {
			return results, fmt.Errorf("failed to process arguments for step %s: %w", step.Name, err)
		}

		// Check if the tool is nil
		if step.Tool == nil {
			return results, fmt.Errorf("step %s has nil tool", step.Name)
		}

		// For tools that need access to previous results (like formatters),
		// pass along the current results as well
		for k, v := range results {
			if _, exists := processedArgs[k]; !exists {
				processedArgs[k] = v
			}
		}

		// Execute the tool
		result, err := step.Tool.Execute(ctx, processedArgs)
		if err != nil {
			if w.errorHandler != nil {
				if handlerErr := w.errorHandler(err); handlerErr != nil {
					return results, fmt.Errorf("step %s failed and error handler also failed: %w", step.Name, handlerErr)
				}
				// Error was handled, continue with the workflow
				continue
			}
			return results, fmt.Errorf("step %s failed: %w", step.Name, err)
		}

		// Store the result
		results[step.Name] = result
	}

	return results, nil
}

// evaluateCondition evaluates a condition expression
// This is a simple implementation that only supports checking if a value exists
func (w *BasicWorkflow) evaluateCondition(condition string, results map[string]interface{}) (bool, error) {
	// Implement a simple condition evaluator
	// For now, just check if a value exists and is truthy

	condition = strings.TrimSpace(condition)

	if strings.HasPrefix(condition, "exists:") {
		key := strings.TrimSpace(strings.TrimPrefix(condition, "exists:"))
		_, exists := results[key]
		return exists, nil
	}

	if strings.HasPrefix(condition, "!exists:") {
		key := strings.TrimSpace(strings.TrimPrefix(condition, "!exists:"))
		_, exists := results[key]
		return !exists, nil
	}

	return false, fmt.Errorf("unsupported condition: %s", condition)
}

// processArgs processes the arguments for a step, replacing templates with values
func (w *BasicWorkflow) processArgs(args map[string]interface{}, results map[string]interface{}) (map[string]interface{}, error) {
	processedArgs := make(map[string]interface{})

	for k, v := range args {
		if strValue, ok := v.(string); ok {
			// Process string templates
			if strings.HasPrefix(strValue, "{{") && strings.HasSuffix(strValue, "}}") {
				// Extract the key
				key := strings.TrimSpace(strings.TrimPrefix(strings.TrimSuffix(strValue, "}}"), "{{"))

				// Split by dots for nested access
				parts := strings.Split(key, ".")

				// Get the value
				value, err := w.getNestedValue(results, parts)
				if err != nil {
					return nil, err
				}

				processedArgs[k] = value
			} else {
				processedArgs[k] = v
			}
		} else {
			processedArgs[k] = v
		}
	}

	return processedArgs, nil
}

// getNestedValue retrieves a nested value from a map
func (w *BasicWorkflow) getNestedValue(data map[string]interface{}, parts []string) (interface{}, error) {
	if len(parts) == 0 {
		return nil, errors.New("empty key")
	}

	key := parts[0]
	value, exists := data[key]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", key)
	}

	if len(parts) == 1 {
		return value, nil
	}

	// Handle nested values
	if nestedMap, ok := value.(map[string]interface{}); ok {
		return w.getNestedValue(nestedMap, parts[1:])
	}

	return nil, fmt.Errorf("cannot access nested value for key: %s", key)
}
