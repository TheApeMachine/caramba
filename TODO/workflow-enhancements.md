# Workflow Orchestration Enhancements

This document outlines a comprehensive plan to enhance the workflow orchestration capabilities of the Caramba framework, focusing on five key areas: conditional branching, parallel execution, workflow visualization, state persistence, and user intervention.

## 1. Conditional Branching

The current workflow system supports basic conditional execution, but we need to enhance it with more sophisticated decision-making capabilities.

### New Files to Create:
- `pkg/agent/workflow/conditions.go`
- `pkg/agent/workflow/expressions.go`

### Implementation Details:

```go
// conditions.go
package workflow

import (
	"context"
	"fmt"
	"reflect"
	"strings"
)

// ConditionEvaluator evaluates conditions for workflow branching
type ConditionEvaluator interface {
	// Evaluate evaluates a condition against the given data
	Evaluate(condition string, data map[string]interface{}) (bool, error)
}

// ExpressionEvaluator evaluates conditional expressions with a mini DSL
type ExpressionEvaluator struct{}

// NewExpressionEvaluator creates a new expression evaluator
func NewExpressionEvaluator() *ExpressionEvaluator {
	return &ExpressionEvaluator{}
}

// Evaluate evaluates a condition using a simple expression language
func (e *ExpressionEvaluator) Evaluate(condition string, data map[string]interface{}) (bool, error) {
	// Handle empty conditions
	if condition == "" {
		return true, nil
	}
	
	// Parse logical operators first (AND, OR)
	if strings.Contains(condition, " AND ") {
		parts := strings.Split(condition, " AND ")
		for _, part := range parts {
			result, err := e.Evaluate(strings.TrimSpace(part), data)
			if err != nil {
				return false, err
			}
			if !result {
				return false, nil // Short-circuit AND
			}
		}
		return true, nil
	}
	
	if strings.Contains(condition, " OR ") {
		parts := strings.Split(condition, " OR ")
		for _, part := range parts {
			result, err := e.Evaluate(strings.TrimSpace(part), data)
			if err != nil {
				return false, err
			}
			if result {
				return true, nil // Short-circuit OR
			}
		}
		return false, nil
	}
	
	// Handle NOT operator
	if strings.HasPrefix(condition, "NOT ") {
		result, err := e.Evaluate(strings.TrimSpace(strings.TrimPrefix(condition, "NOT ")), data)
		if err != nil {
			return false, err
		}
		return !result, nil
	}
	
	// Handle basic comparison operators
	// equals: a == b
	if strings.Contains(condition, " == ") {
		return e.evaluateEquality(condition, data, "==")
	}
	
	// not equals: a != b
	if strings.Contains(condition, " != ") {
		return e.evaluateEquality(condition, data, "!=")
	}
	
	// greater than: a > b
	if strings.Contains(condition, " > ") {
		return e.evaluateComparison(condition, data, ">")
	}
	
	// less than: a < b
	if strings.Contains(condition, " < ") {
		return e.evaluateComparison(condition, data, "<")
	}
	
	// greater than or equal: a >= b
	if strings.Contains(condition, " >= ") {
		return e.evaluateComparison(condition, data, ">=")
	}
	
	// less than or equal: a <= b
	if strings.Contains(condition, " <= ") {
		return e.evaluateComparison(condition, data, "<=")
	}
	
	// contains: a CONTAINS b
	if strings.Contains(condition, " CONTAINS ") {
		return e.evaluateContains(condition, data)
	}
	
	// Handle existence checks
	if strings.HasPrefix(condition, "exists:") {
		key := strings.TrimSpace(strings.TrimPrefix(condition, "exists:"))
		_, exists := getNestedValue(data, strings.Split(key, "."))
		return exists, nil
	}
	
	// Handle non-empty checks
	if strings.HasPrefix(condition, "non_empty:") {
		key := strings.TrimSpace(strings.TrimPrefix(condition, "non_empty:"))
		value, exists := getNestedValue(data, strings.Split(key, "."))
		if !exists {
			return false, nil
		}
		
		// Check if the value is empty
		return !isEmpty(value), nil
	}
	
	// Handle status checks (useful for workflow steps)
	if strings.HasPrefix(condition, "status:") {
		parts := strings.SplitN(strings.TrimPrefix(condition, "status:"), ":", 2)
		if len(parts) != 2 {
			return false, fmt.Errorf("invalid status check format: %s", condition)
		}
		
		stepName := strings.TrimSpace(parts[0])
		status := strings.TrimSpace(parts[1])
		
		stepResult, exists := data[stepName]
		if !exists {
			return false, nil
		}
		
		// If the step result has a status field, check it
		resultMap, ok := stepResult.(map[string]interface{})
		if !ok {
			return false, nil
		}
		
		actualStatus, ok := resultMap["status"].(string)
		if !ok {
			return false, nil
		}
		
		return actualStatus == status, nil
	}
	
	// If we get here, the condition format is not recognized
	return false, fmt.Errorf("unrecognized condition format: %s", condition)
}

// evaluateEquality evaluates equality comparisons
func (e *ExpressionEvaluator) evaluateEquality(condition string, data map[string]interface{}, op string) (bool, error) {
	parts := strings.Split(condition, " "+op+" ")
	if len(parts) != 2 {
		return false, fmt.Errorf("invalid equality expression: %s", condition)
	}
	
	leftOperand := strings.TrimSpace(parts[0])
	rightOperand := strings.TrimSpace(parts[1])
	
	// Resolve left operand
	leftValue, leftErr := resolveOperand(leftOperand, data)
	if leftErr != nil {
		return false, leftErr
	}
	
	// Resolve right operand
	rightValue, rightErr := resolveOperand(rightOperand, data)
	if rightErr != nil {
		return false, rightErr
	}
	
	// Compare based on operator
	if op == "==" {
		return reflect.DeepEqual(leftValue, rightValue), nil
	} else {
		return !reflect.DeepEqual(leftValue, rightValue), nil
	}
}

// evaluateComparison evaluates numeric comparisons
func (e *ExpressionEvaluator) evaluateComparison(condition string, data map[string]interface{}, op string) (bool, error) {
	parts := strings.Split(condition, " "+op+" ")
	if len(parts) != 2 {
		return false, fmt.Errorf("invalid comparison expression: %s", condition)
	}
	
	leftOperand := strings.TrimSpace(parts[0])
	rightOperand := strings.TrimSpace(parts[1])
	
	// Resolve left operand
	leftValue, leftErr := resolveOperand(leftOperand, data)
	if leftErr != nil {
		return false, leftErr
	}
	
	// Resolve right operand
	rightValue, rightErr := resolveOperand(rightOperand, data)
	if rightErr != nil {
		return false, rightErr
	}
	
	// Convert to float64 for comparison
	leftNum, leftOk := toFloat64(leftValue)
	rightNum, rightOk := toFloat64(rightValue)
	
	if !leftOk || !rightOk {
		return false, fmt.Errorf("can't compare non-numeric values: %v %s %v", leftValue, op, rightValue)
	}
	
	// Compare based on operator
	switch op {
	case ">":
		return leftNum > rightNum, nil
	case "<":
		return leftNum < rightNum, nil
	case ">=":
		return leftNum >= rightNum, nil
	case "<=":
		return leftNum <= rightNum, nil
	default:
		return false, fmt.Errorf("unsupported comparison operator: %s", op)
	}
}

// evaluateContains evaluates string or slice contains operation
func (e *ExpressionEvaluator) evaluateContains(condition string, data map[string]interface{}) (bool, error) {
	parts := strings.Split(condition, " CONTAINS ")
	if len(parts) != 2 {
		return false, fmt.Errorf("invalid CONTAINS expression: %s", condition)
	}
	
	leftOperand := strings.TrimSpace(parts[0])
	rightOperand := strings.TrimSpace(parts[1])
	
	// Resolve left operand
	leftValue, leftErr := resolveOperand(leftOperand, data)
	if leftErr != nil {
		return false, leftErr
	}
	
	// Resolve right operand
	rightValue, rightErr := resolveOperand(rightOperand, data)
	if rightErr != nil {
		return false, rightErr
	}
	
	// Handle different container types
	switch container := leftValue.(type) {
	case string:
		searchStr, ok := rightValue.(string)
		if !ok {
			return false, fmt.Errorf("can't check if string contains non-string value")
		}
		return strings.Contains(container, searchStr), nil
		
	case []interface{}:
		for _, item := range container {
			if reflect.DeepEqual(item, rightValue) {
				return true, nil
			}
		}
		return false, nil
		
	case map[string]interface{}:
		key, ok := rightValue.(string)
		if !ok {
			return false, fmt.Errorf("map key must be string")
		}
		_, exists := container[key]
		return exists, nil
		
	default:
		return false, fmt.Errorf("left operand is not a container type: %v", leftValue)
	}
}

// Helper functions

// resolveOperand resolves an operand which can be a literal or a variable reference
func resolveOperand(operand string, data map[string]interface{}) (interface{}, error) {
	// Check if it's a variable reference
	if strings.HasPrefix(operand, "$.") {
		path := strings.TrimPrefix(operand, "$.")
		value, exists := getNestedValue(data, strings.Split(path, "."))
		if !exists {
			return nil, fmt.Errorf("variable not found: %s", path)
		}
		return value, nil
	}
	
	// Otherwise treat as literal
	return parseLiteral(operand), nil
}

// parseLiteral parses a string as a literal value
func parseLiteral(literal string) interface{} {
	// Try to parse as number
	if f, err := strconv.ParseFloat(literal, 64); err == nil {
		return f
	}
	
	// Try to parse as bool
	if b, err := strconv.ParseBool(literal); err == nil {
		return b
	}
	
	// Handle string literals (quoted)
	if (strings.HasPrefix(literal, "\"") && strings.HasSuffix(literal, "\"")) ||
		(strings.HasPrefix(literal, "'") && strings.HasSuffix(literal, "'")) {
		return literal[1 : len(literal)-1]
	}
	
	// Default to treating as string
	return literal
}

// toFloat64 attempts to convert a value to float64
func toFloat64(value interface{}) (float64, bool) {
	switch v := value.(type) {
	case int:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	case float32:
		return float64(v), true
	case float64:
		return v, true
	case string:
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f, true
		}
	}
	return 0, false
}

// isEmpty checks if a value is empty
func isEmpty(value interface{}) bool {
	if value == nil {
		return true
	}
	
	switch v := value.(type) {
	case string:
		return v == ""
	case []interface{}:
		return len(v) == 0
	case map[string]interface{}:
		return len(v) == 0
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		return false // Numbers are never empty
	case bool:
		return !v
	default:
		// For other types, check if it's a zero value
		return reflect.ValueOf(v).IsZero()
	}
}
```

Now, let's update the `BasicWorkflow` struct to support enhanced conditional branching:

```go
// workflow.go
package workflow

// Enhance the Step struct to support more complex conditions
type Step struct {
	Name         string
	Tool         core.Tool
	Args         map[string]interface{}
	Condition    string // Expression to evaluate for conditional execution
	OnSuccess    string // Step to run if this step succeeds
	OnFailure    string // Step to run if this step fails
	Dependencies []string // Steps that must be completed before this step
}

// Enhance the BasicWorkflow struct
type BasicWorkflow struct {
	steps            map[string]Step // Change from slice to map for easier referencing
	entryPoints      []string        // Steps to start with
	errorHandler     func(error) error
	conditionEvaluator ConditionEvaluator
}

// NewWorkflow creates a new workflow
func NewWorkflow() *BasicWorkflow {
	return &BasicWorkflow{
		steps:             make(map[string]Step),
		entryPoints:       []string{},
		conditionEvaluator: NewExpressionEvaluator(),
	}
}

// AddStep adds a step to the workflow
func (w *BasicWorkflow) AddStep(name string, tool core.Tool, args map[string]interface{}) core.Workflow {
	w.steps[name] = Step{
		Name: name,
		Tool: tool,
		Args: args,
	}
	
	// If this is the first step, add it as an entry point
	if len(w.steps) == 1 {
		w.entryPoints = append(w.entryPoints, name)
	}
	
	return w
}

// AddConditionalStep adds a conditional step to the workflow
func (w *BasicWorkflow) AddConditionalStep(name string, condition string, tool core.Tool, args map[string]interface{}) core.Workflow {
	w.steps[name] = Step{
		Name:      name,
		Tool:      tool,
		Args:      args,
		Condition: condition,
	}
	return w
}

// WithCondition adds a condition to the last added step
func (w *BasicWorkflow) WithCondition(condition string) core.Workflow {
	if len(w.steps) == 0 {
		return w
	}
	
	lastStepName := w.entryPoints[len(w.entryPoints)-1]
	step := w.steps[lastStepName]
	step.Condition = condition
	w.steps[lastStepName] = step
	
	return w
}

// OnSuccess specifies the next step to run if this step succeeds
func (w *BasicWorkflow) OnSuccess(stepName string, nextStepName string) core.Workflow {
	if step, exists := w.steps[stepName]; exists {
		step.OnSuccess = nextStepName
		w.steps[stepName] = step
	}
	return w
}

// OnFailure specifies the next step to run if this step fails
func (w *BasicWorkflow) OnFailure(stepName string, nextStepName string) core.Workflow {
	if step, exists := w.steps[stepName]; exists {
		step.OnFailure = nextStepName
		w.steps[stepName] = step
	}
	return w
}

// AddDependency adds a dependency to a step
func (w *BasicWorkflow) AddDependency(stepName string, dependsOn string) core.Workflow {
	if step, exists := w.steps[stepName]; exists {
		step.Dependencies = append(step.Dependencies, dependsOn)
		w.steps[stepName] = step
		
		// Remove from entry points if it was there
		for i, entry := range w.entryPoints {
			if entry == stepName {
				w.entryPoints = append(w.entryPoints[:i], w.entryPoints[i+1:]...)
				break
			}
		}
	}
	return w
}

// Execute executes the workflow with the given input
func (w *BasicWorkflow) Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	results := make(map[string]interface{})
	
	// Add input to results
	for k, v := range input {
		results[k] = v
	}
	
	// Start with entry points
	for _, entryPoint := range w.entryPoints {
		err := w.executeStep(ctx, entryPoint, results)
		if err != nil {
			return results, err
		}
	}
	
	return results, nil
}

// executeStep executes a single step and follows the workflow
func (w *BasicWorkflow) executeStep(ctx context.Context, stepName string, results map[string]interface{}) error {
	step, exists := w.steps[stepName]
	if !exists {
		return fmt.Errorf("step not found: %s", stepName)
	}
	
	// Check if all dependencies are satisfied
	for _, dependency := range step.Dependencies {
		if _, exists := results[dependency]; !exists {
			return fmt.Errorf("dependency not satisfied: %s depends on %s", stepName, dependency)
		}
	}
	
	// Check if the condition allows execution
	if step.Condition != "" {
		shouldExecute, err := w.conditionEvaluator.Evaluate(step.Condition, results)
		if err != nil {
			return fmt.Errorf("failed to evaluate condition for step %s: %w", stepName, err)
		}
		
		if !shouldExecute {
			// Skip this step
			return nil
		}
	}
	
	// Process argument templates
	processedArgs, err := processArgs(step.Args, results)
	if err != nil {
		return fmt.Errorf("failed to process arguments for step %s: %w", stepName, err)
	}
	
	// Check if the tool is nil
	if step.Tool == nil {
		return fmt.Errorf("step %s has nil tool", stepName)
	}
	
	// Execute the tool
	result, err := step.Tool.Execute(ctx, processedArgs)
	
	// Handle success or failure
	if err != nil {
		// Store the error in results
		results[stepName+"_error"] = err.Error()
		
		// Use error handler if available
		if w.errorHandler != nil {
			if handlerErr := w.errorHandler(err); handlerErr != nil {
				return fmt.Errorf("step %s failed and error handler also failed: %w", stepName, handlerErr)
			}
		}
		
		// Follow failure path if defined
		if step.OnFailure != "" {
			return w.executeStep(ctx, step.OnFailure, results)
		}
		
		// Otherwise propagate the error
		return fmt.Errorf("step %s failed: %w", stepName, err)
	}
	
	// Store the result
	results[stepName] = result
	
	// Follow success path if defined
	if step.OnSuccess != "" {
		return w.executeStep(ctx, step.OnSuccess, results)
	}
	
	return nil
}
```

## 2. Parallel Execution

To support concurrent execution of workflow steps, we'll add parallel execution capabilities to the workflow system.

### New Files to Create:
- `pkg/agent/workflow/parallel.go`

### Implementation Details:

```go
// parallel.go
package workflow

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ParallelStepGroup represents a group of steps to be executed in parallel
type ParallelStepGroup struct {
	Steps       []string  // Steps to execute in parallel
	MaxConcurrent int     // Maximum number of concurrent steps (0 for unlimited)
	Timeout     time.Duration // Timeout for the entire group
	AllMustSucceed bool   // If true, any failure fails the entire group
}

// ParallelWorkflow extends BasicWorkflow with parallel execution capabilities
type ParallelWorkflow struct {
	*BasicWorkflow
	parallelGroups map[string]ParallelStepGroup
}

// NewParallelWorkflow creates a new workflow with parallel execution support
func NewParallelWorkflow() *ParallelWorkflow {
	return &ParallelWorkflow{
		BasicWorkflow:  NewWorkflow(),
		parallelGroups: make(map[string]ParallelStepGroup),
	}
}

// AddParallelGroup adds a group of steps to be executed in parallel
func (w *ParallelWorkflow) AddParallelGroup(name string, steps []string, options ...ParallelOption) core.Workflow {
	group := ParallelStepGroup{
		Steps:         steps,
		MaxConcurrent: 0,       // Default: unlimited
		Timeout:       0,       // Default: no timeout
		AllMustSucceed: true,   // Default: all must succeed
	}
	
	// Apply options
	for _, option := range options {
		option(&group)
	}
	
	w.parallelGroups[name] = group
	
	// Add a synthetic step that represents the parallel group
	w.AddStep(name, NewParallelGroupTool(w, name), nil)
	
	return w
}

// ParallelOption is a functional option for configuring parallel execution
type ParallelOption func(*ParallelStepGroup)

// WithMaxConcurrent sets the maximum number of concurrent steps
func WithMaxConcurrent(max int) ParallelOption {
	return func(g *ParallelStepGroup) {
		g.MaxConcurrent = max
	}
}

// WithTimeout sets a timeout for the parallel group
func WithTimeout(timeout time.Duration) ParallelOption {
	return func(g *ParallelStepGroup) {
		g.Timeout = timeout
	}
}

// WithAnySuccess configures the group to succeed if any step succeeds
func WithAnySuccess() ParallelOption {
	return func(g *ParallelStepGroup) {
		g.AllMustSucceed = false
	}
}

// ParallelGroupTool is a synthetic tool that executes steps in parallel
type ParallelGroupTool struct {
	workflow *ParallelWorkflow
	groupName string
}

// NewParallelGroupTool creates a new parallel group execution tool
func NewParallelGroupTool(workflow *ParallelWorkflow, groupName string) *ParallelGroupTool {
	return &ParallelGroupTool{
		workflow: workflow,
		groupName: groupName,
	}
}

// Name returns the name of the tool
func (t *ParallelGroupTool) Name() string {
	return "parallel_executor"
}

// Description returns the description of the tool
func (t *ParallelGroupTool) Description() string {
	return "Executes multiple workflow steps in parallel"
}

// Schema returns the JSON schema for the tool's arguments
func (t *ParallelGroupTool) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"timeout_ms": map[string]interface{}{
				"type": "integer",
				"description": "Optional timeout in milliseconds",
			},
		},
	}
}

// Execute executes the parallel group tool
func (t *ParallelGroupTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	group, exists := t.workflow.parallelGroups[t.groupName]
	if !exists {
		return nil, fmt.Errorf("parallel group not found: %s", t.groupName)
	}
	
	// Override timeout if specified in args
	if timeoutMs, ok := args["timeout_ms"].(int); ok && timeoutMs > 0 {
		group.Timeout = time.Duration(timeoutMs) * time.Millisecond
	}
	
	// Create a context with timeout if specified
	execCtx := ctx
	var cancel context.CancelFunc
	if group.Timeout > 0 {
		execCtx, cancel = context.WithTimeout(ctx, group.Timeout)
		defer cancel()
	}
	
	// Prepare a semaphore for concurrency control
	var sem chan struct{}
	if group.MaxConcurrent > 0 {
		sem = make(chan struct{}, group.MaxConcurrent)
	}
	
	// Create a wait group to wait for all steps
	var wg sync.WaitGroup
	
	// Prepare results collection
	resultsMu := sync.Mutex{}
	stepResults := make(map[string]interface{})
	stepErrors := make(map[string]error)
	
	// Execute steps in parallel
	for _, stepName := range group.Steps {
		wg.Add(1)
		
		// Get the step
		step, exists := t.workflow.steps[stepName]
		if !exists {
			return nil, fmt.Errorf("step not found: %s", stepName)
		}
		
		// Process arguments
		processedArgs, err := processArgs(step.Args, args)
		if err != nil {
			return nil, fmt.Errorf("failed to process arguments for step %s: %w", stepName, err)
		}
		
		// Launch the step in a goroutine
		go func(name string, step Step, args map[string]interface{}) {
			defer wg.Done()
			
			// Acquire semaphore slot if needed
			if sem != nil {
				select {
				case sem <- struct{}{}:
					defer func() { <-sem }()
				case <-execCtx.Done():
					resultsMu.Lock()
					stepErrors[name] = execCtx.Err()
					resultsMu.Unlock()
					return
				}
			}
			
			// Execute the step
			result, err := step.Tool.Execute(execCtx, args)
			
			// Store the result
			resultsMu.Lock()
			if err != nil {
				stepErrors[name] = err
			} else {
				stepResults[name] = result
			}
			resultsMu.Unlock()
		}(stepName, step, processedArgs)
	}
	
	// Wait for all steps to complete
	wg.Wait()
	
	// Check results based on AllMustSucceed flag
	if group.AllMustSucceed && len(stepErrors) > 0 {
		// Combine all errors
		errorMsg := "parallel execution failed:"
		for step, err := range stepErrors {
			errorMsg += fmt.Sprintf("\n  - %s: %v", step, err)
		}
		return stepResults, fmt.Errorf(errorMsg)
	}
	
	// Return combined results
	return map[string]interface{}{
		"results": stepResults,
		"errors":  stepErrors,
		"success": len(stepErrors) == 0 || !group.AllMustSucceed,
	}, nil
}
```

Now, let's update the main `workflow.go` file to include parallel execution support:

```go
// In workflow.go, extend the Workflow interface

// Workflow defines the interface for a workflow
type Workflow interface {
	// AddStep adds a step to the workflow
	AddStep(name string, tool Tool, args map[string]interface{}) Workflow

	// AddConditionalStep adds a conditional step to the workflow
	AddConditionalStep(name string, condition string, tool Tool, args map[string]interface{}) Workflow

	// Execute executes the workflow with the given input
	Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error)

	// SetErrorHandler sets a handler for errors
	SetErrorHandler(handler func(error) error) Workflow
	
	// New methods for conditional flow
	WithCondition(condition string) Workflow
	OnSuccess(stepName string, nextStepName string) Workflow
	OnFailure(stepName string, nextStepName string) Workflow
	AddDependency(stepName string, dependsOn string) Workflow
	
	// New methods for parallel execution
	AddParallelGroup(name string, steps []string, options ...ParallelOption) Workflow
}
```

## 3. Workflow Visualization

To visualize complex workflows, we'll implement a visualization tool that can generate diagrams in various formats including Mermaid, DOT (Graphviz), and simple ASCII representations.

### New Files to Create:
- `pkg/agent/workflow/visualization.go`

### Implementation Details:

```go
// visualization.go
package workflow

import (
	"bytes"
	"fmt"
	"io"
	"strings"
)

// Format specifies the output format for workflow visualization
type Format string

const (
	FormatMermaid Format = "mermaid"
	FormatDOT     Format = "dot"
	FormatASCII   Format = "ascii"
)

// Visualizer creates visual representations of workflows
type Visualizer interface {
	// Visualize renders a workflow visualization
	Visualize(workflow core.Workflow, format Format) (string, error)
	
	// SaveVisualization saves a workflow visualization to a file
	SaveVisualization(workflow core.Workflow, format Format, fileName string) error
}

// WorkflowVisualizer implements the Visualizer interface
type WorkflowVisualizer struct{}

// NewWorkflowVisualizer creates a new workflow visualizer
func NewWorkflowVisualizer() *WorkflowVisualizer {
	return &WorkflowVisualizer{}
}

// Visualize renders a workflow visualization
func (v *WorkflowVisualizer) Visualize(workflow core.Workflow, format Format) (string, error) {
	switch format {
	case FormatMermaid:
		return v.generateMermaid(workflow)
	case FormatDOT:
		return v.generateDOT(workflow)
	case FormatASCII:
		return v.generateASCII(workflow)
	default:
		return "", fmt.Errorf("unsupported visualization format: %s", format)
	}
}

// SaveVisualization saves a workflow visualization to a file
func (v *WorkflowVisualizer) SaveVisualization(workflow core.Workflow, format Format, fileName string) error {
	content, err := v.Visualize(workflow, format)
	if err != nil {
		return err
	}
	
	return ioutil.WriteFile(fileName, []byte(content), 0644)
}

// generateMermaid generates a workflow visualization in Mermaid format
func (v *WorkflowVisualizer) generateMermaid(workflow core.Workflow) (string, error) {
	// Check if the workflow implements the necessary interfaces
	basicWorkflow, ok := workflow.(*BasicWorkflow)
	if !ok {
		// Try to get BasicWorkflow from ParallelWorkflow
		if parallelWorkflow, ok := workflow.(*ParallelWorkflow); ok {
			basicWorkflow = parallelWorkflow.BasicWorkflow
		} else {
			return "", fmt.Errorf("unsupported workflow type for visualization")
		}
	}
	
	var buffer bytes.Buffer
	buffer.WriteString("flowchart TD\n")
	
	// Create nodes for each step
	for name, step := range basicWorkflow.steps {
		// Generate node representation
		nodeID := sanitizeID(name)
		nodeLabel := name
		if step.Tool != nil {
			nodeLabel = fmt.Sprintf("%s(%s)", name, step.Tool.Name())
		}
		
		buffer.WriteString(fmt.Sprintf("    %s[\"%s\"]\n", nodeID, nodeLabel))
	}
	
	// Draw connections
	for name, step := range basicWorkflow.steps {
		fromID := sanitizeID(name)
		
		// Normal success flow
		if step.OnSuccess != "" {
			toID := sanitizeID(step.OnSuccess)
			buffer.WriteString(fmt.Sprintf("    %s -->|success| %s\n", fromID, toID))
		}
		
		// Error flow
		if step.OnFailure != "" {
			toID := sanitizeID(step.OnFailure)
			buffer.WriteString(fmt.Sprintf("    %s -->|failure| %s\n", fromID, toID))
		}
		
		// Dependencies
		for _, dep := range step.Dependencies {
			depID := sanitizeID(dep)
			buffer.WriteString(fmt.Sprintf("    %s -->|depends on| %s\n", depID, fromID))
		}
	}
	
	// Handle parallel groups
	if parallelWorkflow, ok := workflow.(*ParallelWorkflow); ok {
		for groupName, group := range parallelWorkflow.parallelGroups {
			groupID := sanitizeID("group_" + groupName)
			buffer.WriteString(fmt.Sprintf("    subgraph %s[%s]\n", groupID, groupName))
			
			// Add nodes for steps in the group
			for _, stepName := range group.Steps {
				stepID := sanitizeID(stepName)
				buffer.WriteString(fmt.Sprintf("        %s\n", stepID))
			}
			
			buffer.WriteString("    end\n")
			
			// Connect the group to its steps with style
			buffer.WriteString(fmt.Sprintf("    linkStyle default stroke-width:2,fill:none,stroke:blue\n"))
		}
	}
	
	return buffer.String(), nil
}

// generateDOT generates a workflow visualization in Graphviz DOT format
func (v *WorkflowVisualizer) generateDOT(workflow core.Workflow) (string, error) {
	// Check if the workflow implements the necessary interfaces
	basicWorkflow, ok := workflow.(*BasicWorkflow)
	if !ok {
		// Try to get BasicWorkflow from ParallelWorkflow
		if parallelWorkflow, ok := workflow.(*ParallelWorkflow); ok {
			basicWorkflow = parallelWorkflow.BasicWorkflow
		} else {
			return "", fmt.Errorf("unsupported workflow type for visualization")
		}
	}
	
	var buffer bytes.Buffer
	buffer.WriteString("digraph Workflow {\n")
	buffer.WriteString("    node [shape=box, style=filled, fillcolor=lightblue];\n")
	buffer.WriteString("    edge [fontsize=10];\n")
	
	// Create nodes for each step
	for name, step := range basicWorkflow.steps {
		// Generate node representation
		nodeID := sanitizeID(name)
		nodeLabel := name
		if step.Tool != nil {
			nodeLabel = fmt.Sprintf("%s\\n(%s)", name, step.Tool.Name())
		}
		
		buffer.WriteString(fmt.Sprintf("    %s [label=\"%s\"];\n", nodeID, nodeLabel))
	}
	
	// Draw connections
	for name, step := range basicWorkflow.steps {
		fromID := sanitizeID(name)
		
		// Normal success flow
		if step.OnSuccess != "" {
			toID := sanitizeID(step.OnSuccess)
			buffer.WriteString(fmt.Sprintf("    %s -> %s [label=\"success\", color=green];\n", fromID, toID))
		}
		
		// Error flow
		if step.OnFailure != "" {
			toID := sanitizeID(step.OnFailure)
			buffer.WriteString(fmt.Sprintf("    %s -> %s [label=\"failure\", color=red];\n", fromID, toID))
		}
		
		// Dependencies
		for _, dep := range step.Dependencies {
			depID := sanitizeID(dep)
			buffer.WriteString(fmt.Sprintf("    %s -> %s [label=\"depends on\", style=dashed];\n", depID, fromID))
		}
	}
	
	// Handle parallel groups
	if parallelWorkflow, ok := workflow.(*ParallelWorkflow); ok {
		for groupName, group := range parallelWorkflow.parallelGroups {
			groupID := sanitizeID("group_" + groupName)
			buffer.WriteString(fmt.Sprintf("    subgraph cluster_%s {\n", groupID))
			buffer.WriteString(fmt.Sprintf("        label=\"%s\";\n", groupName))
			buffer.WriteString("        style=filled;\n")
			buffer.WriteString("        color=lightgrey;\n")
			
			// Add nodes for steps in the group
			for _, stepName := range group.Steps {
				stepID := sanitizeID(stepName)
				buffer.WriteString(fmt.Sprintf("        %s;\n", stepID))
			}
			
			buffer.WriteString("    }\n")
		}
	}
	
	buffer.WriteString("}\n")
	return buffer.String(), nil
}

// generateASCII generates a simple ASCII representation of the workflow
func (v *WorkflowVisualizer) generateASCII(workflow core.Workflow) (string, error) {
	// Check if the workflow implements the necessary interfaces
	basicWorkflow, ok := workflow.(*BasicWorkflow)
	if !ok {
		// Try to get BasicWorkflow from ParallelWorkflow
		if parallelWorkflow, ok := workflow.(*ParallelWorkflow); ok {
			basicWorkflow = parallelWorkflow.BasicWorkflow
		} else {
			return "", fmt.Errorf("unsupported workflow type for visualization")
		}
	}
	
	var buffer bytes.Buffer
	buffer.WriteString("Workflow Structure:\n")
	buffer.WriteString("=================\n\n")
	
	// List all steps
	buffer.WriteString("Steps:\n")
	for name, step := range basicWorkflow.steps {
		toolName := "unknown"
		if step.Tool != nil {
			toolName = step.Tool.Name()
		}
		
		buffer.WriteString(fmt.Sprintf("- %s (%s)\n", name, toolName))
		
		// Show conditions
		if step.Condition != "" {
			buffer.WriteString(fmt.Sprintf("  Condition: %s\n", step.Condition))
		}
		
		// Show dependencies
		if len(step.Dependencies) > 0 {
			buffer.WriteString(fmt.Sprintf("  Dependencies: %s\n", strings.Join(step.Dependencies, ", ")))
		}
		
		// Show next steps
		if step.OnSuccess != "" {
			buffer.WriteString(fmt.Sprintf("  On Success: %s\n", step.OnSuccess))
		}
		if step.OnFailure != "" {
			buffer.WriteString(fmt.Sprintf("  On Failure: %s\n", step.OnFailure))
		}
		
		buffer.WriteString("\n")
	}
	
	// Show parallel groups
	if parallelWorkflow, ok := workflow.(*ParallelWorkflow); ok && len(parallelWorkflow.parallelGroups) > 0 {
		buffer.WriteString("Parallel Groups:\n")
		for name, group := range parallelWorkflow.parallelGroups {
			buffer.WriteString(fmt.Sprintf("- %s\n", name))
			buffer.WriteString(fmt.Sprintf("  Steps: %s\n", strings.Join(group.Steps, ", ")))
			
			if group.MaxConcurrent > 0 {
				buffer.WriteString(fmt.Sprintf("  Max Concurrent: %d\n", group.MaxConcurrent))
			}
			
			if group.Timeout > 0 {
				buffer.WriteString(fmt.Sprintf("  Timeout: %s\n", group.Timeout))
			}
			
			buffer.WriteString(fmt.Sprintf("  All Must Succeed: %v\n", group.AllMustSucceed))
			buffer.WriteString("\n")
		}
	}
	
	return buffer.String(), nil
}

// Helper functions
func sanitizeID(id string) string {
	// Replace characters that might cause issues in graph formats
	replacer := strings.NewReplacer(
		" ", "_",
		"-", "_",
		".", "_",
		",", "_",
		"(", "_",
		")", "_",
		"[", "_",
		"]", "_",
		"{", "_",
		"}", "_",
		"\"", "_",
		"'", "_",
		":", "_",
		";", "_",
		"+", "_plus_",
		"*", "_star_",
		"/", "_div_",
		"\\", "_bslash_",
		"<", "_lt_",
		">", "_gt_",
		"&", "_amp_",
		"|", "_pipe_",
		"!", "_excl_",
		"?", "_quest_",
		"=", "_eq_",
		"$", "_dollar_",
		"#", "_hash_",
		"@", "_at_",
		"%", "_percent_",
		"^", "_caret_",
		"~", "_tilde_",
		"`", "_backtick_",
	)
	
	return replacer.Replace(id)
}