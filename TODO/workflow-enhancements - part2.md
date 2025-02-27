	RequestID  string                 `json:"request_id"`
	Approved   bool                   `json:"approved,omitempty"`   // For approvals
	Decision   string                 `json:"decision,omitempty"`   // For decisions
	Input      map[string]interface{} `json:"input,omitempty"`      // For inputs
	Comment    string                 `json:"comment,omitempty"`    // Optional comment
	RespondedAt time.Time             `json:"responded_at"`
	RespondedBy string                 `json:"responded_by,omitempty"`
}

// InterventionHandler manages user intervention requests and responses
type InterventionHandler interface {
	// RequestApproval requests user approval to proceed
	RequestApproval(ctx context.Context, workflowID, stepName, title, description string) (*InterventionResponse, error)
	
	// RequestInput requests input from the user
	RequestInput(ctx context.Context, workflowID, stepName, title, description string, schema interface{}) (*InterventionResponse, error)
	
	// RequestDecision requests a decision from the user
	RequestDecision(ctx context.Context, workflowID, stepName, title, description string, options []string) (*InterventionResponse, error)
	
	// GetPendingRequests returns all pending intervention requests
	GetPendingRequests(ctx context.Context, workflowID string) ([]*InterventionRequest, error)
	
	// RespondToRequest submits a response to an intervention request
	RespondToRequest(ctx context.Context, response *InterventionResponse) error
}

// ChannelBasedInterventionHandler implements InterventionHandler using channels
type ChannelBasedInterventionHandler struct {
	requests      map[string]*InterventionRequest
	responses     map[string]chan *InterventionResponse
	pendingByWorkflow map[string][]*InterventionRequest
	mu           sync.RWMutex
	timeout      time.Duration
}

// NewChannelBasedInterventionHandler creates a new channel-based intervention handler
func NewChannelBasedInterventionHandler(timeout time.Duration) *ChannelBasedInterventionHandler {
	if timeout <= 0 {
		timeout = 24 * time.Hour // Default timeout
	}
	
	return &ChannelBasedInterventionHandler{
		requests:     make(map[string]*InterventionRequest),
		responses:    make(map[string]chan *InterventionResponse),
		pendingByWorkflow: make(map[string][]*InterventionRequest),
		timeout:      timeout,
	}
}

// RequestApproval requests user approval to proceed
func (h *ChannelBasedInterventionHandler) RequestApproval(
	ctx context.Context, 
	workflowID, 
	stepName, 
	title, 
	description string,
) (*InterventionResponse, error) {
	// Create request
	requestID := fmt.Sprintf("approval-%s-%s-%d", workflowID, stepName, time.Now().UnixNano())
	expiresAt := time.Now().Add(h.timeout)
	
	request := &InterventionRequest{
		ID:          requestID,
		WorkflowID:  workflowID,
		StepName:    stepName,
		Type:        InterventionTypeApproval,
		Title:       title,
		Description: description,
		CreatedAt:   time.Now(),
		ExpiresAt:   &expiresAt,
	}
	
	// Create response channel
	responseCh := make(chan *InterventionResponse, 1)
	
	// Store the request and response channel
	h.mu.Lock()
	h.requests[requestID] = request
	h.responses[requestID] = responseCh
	
	// Add to pending requests for this workflow
	h.pendingByWorkflow[workflowID] = append(h.pendingByWorkflow[workflowID], request)
	h.mu.Unlock()
	
	// Wait for response or timeout
	timeoutCtx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()
	
	select {
	case response := <-responseCh:
		return response, nil
	case <-timeoutCtx.Done():
		// Clean up
		h.mu.Lock()
		delete(h.responses, requestID)
		
		// Mark as expired in the requests map but don't delete
		// so the UI can still show it as expired
		if req, exists := h.requests[requestID]; exists {
			now := time.Now()
			req.ExpiresAt = &now
		}
		
		// Remove from pending
		if pending, exists := h.pendingByWorkflow[workflowID]; exists {
			filtered := make([]*InterventionRequest, 0, len(pending))
			for _, req := range pending {
				if req.ID != requestID {
					filtered = append(filtered, req)
				}
			}
			h.pendingByWorkflow[workflowID] = filtered
		}
		h.mu.Unlock()
		
		return nil, fmt.Errorf("approval request timed out")
	}
}

// RequestInput requests input from the user
func (h *ChannelBasedInterventionHandler) RequestInput(
	ctx context.Context, 
	workflowID, 
	stepName, 
	title, 
	description string, 
	schema interface{},
) (*InterventionResponse, error) {
	// Create request
	requestID := fmt.Sprintf("input-%s-%s-%d", workflowID, stepName, time.Now().UnixNano())
	expiresAt := time.Now().Add(h.timeout)
	
	request := &InterventionRequest{
		ID:          requestID,
		WorkflowID:  workflowID,
		StepName:    stepName,
		Type:        InterventionTypeInput,
		Title:       title,
		Description: description,
		InputSchema: schema,
		CreatedAt:   time.Now(),
		ExpiresAt:   &expiresAt,
	}
	
	// Create response channel
	responseCh := make(chan *InterventionResponse, 1)
	
	// Store the request and response channel
	h.mu.Lock()
	h.requests[requestID] = request
	h.responses[requestID] = responseCh
	
	// Add to pending requests for this workflow
	h.pendingByWorkflow[workflowID] = append(h.pendingByWorkflow[workflowID], request)
	h.mu.Unlock()
	
	// Wait for response or timeout
	timeoutCtx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()
	
	select {
	case response := <-responseCh:
		return response, nil
	case <-timeoutCtx.Done():
		// Clean up
		h.mu.Lock()
		delete(h.responses, requestID)
		
		// Mark as expired
		if req, exists := h.requests[requestID]; exists {
			now := time.Now()
			req.ExpiresAt = &now
		}
		
		// Remove from pending
		if pending, exists := h.pendingByWorkflow[workflowID]; exists {
			filtered := make([]*InterventionRequest, 0, len(pending))
			for _, req := range pending {
				if req.ID != requestID {
					filtered = append(filtered, req)
				}
			}
			h.pendingByWorkflow[workflowID] = filtered
		}
		h.mu.Unlock()
		
		return nil, fmt.Errorf("input request timed out")
	}
}

// RequestDecision requests a decision from the user
func (h *ChannelBasedInterventionHandler) RequestDecision(
	ctx context.Context, 
	workflowID, 
	stepName, 
	title, 
	description string, 
	options []string,
) (*InterventionResponse, error) {
	// Create request
	requestID := fmt.Sprintf("decision-%s-%s-%d", workflowID, stepName, time.Now().UnixNano())
	expiresAt := time.Now().Add(h.timeout)
	
	request := &InterventionRequest{
		ID:          requestID,
		WorkflowID:  workflowID,
		StepName:    stepName,
		Type:        InterventionTypeDecision,
		Title:       title,
		Description: description,
		Options:     options,
		CreatedAt:   time.Now(),
		ExpiresAt:   &expiresAt,
	}
	
	// Create response channel
	responseCh := make(chan *InterventionResponse, 1)
	
	// Store the request and response channel
	h.mu.Lock()
	h.requests[requestID] = request
	h.responses[requestID] = responseCh
	
	// Add to pending requests for this workflow
	h.pendingByWorkflow[workflowID] = append(h.pendingByWorkflow[workflowID], request)
	h.mu.Unlock()
	
	// Wait for response or timeout
	timeoutCtx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()
	
	select {
	case response := <-responseCh:
		return response, nil
	case <-timeoutCtx.Done():
		// Clean up
		h.mu.Lock()
		delete(h.responses, requestID)
		
		// Mark as expired
		if req, exists := h.requests[requestID]; exists {
			now := time.Now()
			req.ExpiresAt = &now
		}
		
		// Remove from pending
		if pending, exists := h.pendingByWorkflow[workflowID]; exists {
			filtered := make([]*InterventionRequest, 0, len(pending))
			for _, req := range pending {
				if req.ID != requestID {
					filtered = append(filtered, req)
				}
			}
			h.pendingByWorkflow[workflowID] = filtered
		}
		h.mu.Unlock()
		
		return nil, fmt.Errorf("decision request timed out")
	}
}

// GetPendingRequests returns all pending intervention requests
func (h *ChannelBasedInterventionHandler) GetPendingRequests(ctx context.Context, workflowID string) ([]*InterventionRequest, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	
	if workflowID == "" {
		// Return all pending requests from all workflows
		var allPending []*InterventionRequest
		for _, pending := range h.pendingByWorkflow {
			allPending = append(allPending, pending...)
		}
		return allPending, nil
	}
	
	// Return pending requests for the specified workflow
	if pending, exists := h.pendingByWorkflow[workflowID]; exists {
		// Make a copy to avoid race conditions
		result := make([]*InterventionRequest, len(pending))
		copy(result, pending)
		return result, nil
	}
	
	return nil, nil
}

// RespondToRequest submits a response to an intervention request
func (h *ChannelBasedInterventionHandler) RespondToRequest(ctx context.Context, response *InterventionResponse) error {
	h.mu.RLock()
	request, requestExists := h.requests[response.RequestID]
	responseCh, responseExists := h.responses[response.RequestID]
	h.mu.RUnlock()
	
	if !requestExists {
		return fmt.Errorf("intervention request not found: %s", response.RequestID)
	}
	
	if !responseExists {
		return fmt.Errorf("intervention request expired or already responded to: %s", response.RequestID)
	}
	
	// Validate response based on request type
	if err := validateResponse(request, response); err != nil {
		return err
	}
	
	// Set response time
	response.RespondedAt = time.Now()
	
	// Submit the response
	select {
	case responseCh <- response:
		// Successfully submitted
	default:
		return fmt.Errorf("response channel full or closed")
	}
	
	// Update pending requests
	h.mu.Lock()
	defer h.mu.Unlock()
	
	// Remove from pending
	if pending, exists := h.pendingByWorkflow[request.WorkflowID]; exists {
		filtered := make([]*InterventionRequest, 0, len(pending))
		for _, req := range pending {
			if req.ID != response.RequestID {
				filtered = append(filtered, req)
			}
		}
		h.pendingByWorkflow[request.WorkflowID] = filtered
	}
	
	// Remove the response channel
	delete(h.responses, response.RequestID)
	
	return nil
}

// validateResponse validates a response against the request type
func validateResponse(request *InterventionRequest, response *InterventionResponse) error {
	switch request.Type {
	case InterventionTypeApproval:
		// For approvals, the 'Approved' field must be set
		// No further validation needed
		
	case InterventionTypeDecision:
		// For decisions, check if the decision is valid
		if response.Decision == "" {
			return fmt.Errorf("decision response missing decision value")
		}
		
		valid := false
		for _, option := range request.Options {
			if option == response.Decision {
				valid = true
				break
			}
		}
		
		if !valid {
			return fmt.Errorf("invalid decision: %s (valid options: %v)", response.Decision, request.Options)
		}
		
	case InterventionTypeInput:
		// For inputs, check if there is input data
		if response.Input == nil {
			return fmt.Errorf("input response missing input data")
		}
		
		// TODO: Validate input against schema if provided
	}
	
	return nil
}
```

Now, let's create a user intervention tool that can be used in workflows:

```go
// Add to workflow.go

// UserInterventionTool is a tool for requesting user intervention
type UserInterventionTool struct {
	interventionHandler InterventionHandler
}

// NewUserInterventionTool creates a new user intervention tool
func NewUserInterventionTool(handler InterventionHandler) *UserInterventionTool {
	return &UserInterventionTool{
		interventionHandler: handler,
	}
}

// Name returns the name of the tool
func (t *UserInterventionTool) Name() string {
	return "user_intervention"
}

// Description returns the description of the tool
func (t *UserInterventionTool) Description() string {
	return "Requests intervention from a human user"
}

// Schema returns the JSON schema for the tool's arguments
func (t *UserInterventionTool) Schema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"intervention_type": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"approval", "input", "decision"},
				"description": "Type of intervention to request",
			},
			"workflow_id": map[string]interface{}{
				"type":        "string",
				"description": "ID of the workflow (will be auto-populated)",
			},
			"step_name": map[string]interface{}{
				"type":        "string",
				"description": "Name of the step (will be auto-populated)",
			},
			"title": map[string]interface{}{
				"type":        "string",
				"description": "Title of the intervention request",
			},
			"description": map[string]interface{}{
				"type":        "string",
				"description": "Detailed description of what intervention is needed",
			},
			"options": map[string]interface{}{
				"type":        "array",
				"items":       map[string]interface{}{"type": "string"},
				"description": "Options for decision-type interventions",
			},
			"input_schema": map[string]interface{}{
				"type":        "object",
				"description": "JSON Schema for input-type interventions",
			},
		},
		"required": []string{"intervention_type", "title", "description"},
	}
}

// Execute executes the tool with the given arguments
func (t *UserInterventionTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	// Extract common arguments
	interventionType, _ := args["intervention_type"].(string)
	workflowID, _ := args["workflow_id"].(string)
	stepName, _ := args["step_name"].(string)
	title, _ := args["title"].(string)
	description, _ := args["description"].(string)
	
	if title == "" || description == "" {
		return nil, fmt.Errorf("title and description are required")
	}
	
	var response *InterventionResponse
	var err error
	
	switch interventionType {
	case "approval":
		response, err = t.interventionHandler.RequestApproval(ctx, workflowID, stepName, title, description)
		
	case "input":
		inputSchema, _ := args["input_schema"].(map[string]interface{})
		response, err = t.interventionHandler.RequestInput(ctx, workflowID, stepName, title, description, inputSchema)
		
	case "decision":
		optionsAny, ok := args["options"].([]interface{})
		if !ok {
			return nil, fmt.Errorf("options must be an array of strings")
		}
		
		options := make([]string, len(optionsAny))
		for i, o := range optionsAny {
			options[i], _ = o.(string)
		}
		
		response, err = t.interventionHandler.RequestDecision(ctx, workflowID, stepName, title, description, options)
		
	default:
		return nil, fmt.Errorf("invalid intervention type: %s", interventionType)
	}
	
	if err != nil {
		return nil, err
	}
	
	// Convert response to a map for the result
	result := map[string]interface{}{
		"request_id": response.RequestID,
		"responded_at": response.RespondedAt,
	}
	
	switch interventionType {
	case "approval":
		result["approved"] = response.Approved
	case "decision":
		result["decision"] = response.Decision
	case "input":
		result["input"] = response.Input
	}
	
	if response.Comment != "" {
		result["comment"] = response.Comment
	}
	
	if response.RespondedBy != "" {
		result["responded_by"] = response.RespondedBy
	}
	
	return result, nil
}
```

## Integration with the Workflow System

To integrate all these enhancements into the workflow system, we need to update the `core.Workflow` interface in `pkg/agent/core/interfaces.go`:

```go
// Workflow defines the interface for a workflow.
type Workflow interface {
	// AddStep adds a step to the workflow
	AddStep(name string, tool Tool, args map[string]interface{}) Workflow

	// AddConditionalStep adds a conditional step to the workflow
	AddConditionalStep(name string, condition string, tool Tool, args map[string]interface{}) Workflow

	// Execute executes the workflow with the given input
	Execute(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error)

	// SetErrorHandler sets a handler for errors
	SetErrorHandler(handler func(error) error) Workflow
	
	// WithCondition adds a condition to the last added step
	WithCondition(condition string) Workflow
	
	// OnSuccess specifies the next step to run if this step succeeds
	OnSuccess(stepName string, nextStepName string) Workflow
	
	// OnFailure specifies the next step to run if this step fails
	OnFailure(stepName string, nextStepName string) Workflow
	
	// AddDependency adds a dependency to a step
	AddDependency(stepName string, dependsOn string) Workflow
	
	// AddParallelGroup adds a group of steps to be executed in parallel
	AddParallelGroup(name string, steps []string, options ...ParallelOption) Workflow
	
	// Visualize renders a visualization of the workflow
	Visualize(format Format) (string, error)
}

// CheckpointableWorkflow extends the Workflow interface with checkpointing capabilities
type CheckpointableWorkflow interface {
	Workflow
	
	// SaveCheckpoint saves the current state of the workflow
	SaveCheckpoint(ctx context.Context) (string, error)
	
	// ResumeFromCheckpoint resumes execution from a saved checkpoint
	ResumeFromCheckpoint(ctx context.Context, checkpointID string) (map[string]interface{}, error)
	
	// WithPersistence configures the persistence backend
	WithPersistence(persistence WorkflowPersistence) CheckpointableWorkflow
}

// InteractiveWorkflow extends the Workflow interface with user interaction capabilities
type InteractiveWorkflow interface {
	Workflow
	
	// WithInterventionHandler sets the intervention handler for the workflow
	WithInterventionHandler(handler InterventionHandler) InteractiveWorkflow
}
```

## Example Usage

Here's an example of how to use these enhanced workflow capabilities:

```go
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/theapemachine/caramba/pkg/agent/core"
	"github.com/theapemachine/caramba/pkg/agent/tools"
	"github.com/theapemachine/caramba/pkg/agent/workflow"
)

func main() {
	// Create a basic calculator tool for demonstration
	calculator := tools.NewCalculator()
	
	// Create an intervention handler
	interventionHandler := workflow.NewChannelBasedInterventionHandler(1 * time.Hour)
	
	// Create a user intervention tool
	userInterventionTool := workflow.NewUserInterventionTool(interventionHandler)
	
	// Create a workflow with checkpointing and persistence
	wf := workflow.NewCheckpointableWorkflow("Math Workflow")
	
	// Set up persistence
	persistence, err := workflow.NewFileSystemPersistence("./workflow_states")
	if err != nil {
		fmt.Printf("Error setting up persistence: %v\n", err)
		os.Exit(1)
	}
	
	wf.WithPersistence(persistence)
	
	// Define the workflow steps with branches and conditions
	wf.AddStep("start", calculator, map[string]interface{}{
		"expression": "10 + 5",
	})
	
	wf.AddStep("check_result", userInterventionTool, map[string]interface{}{
		"intervention_type": "decision",
		"title":            "Check First Calculation Result",
		"description":      "The first calculation result is: {{.start}}. How would you like to proceed?",
		"options":          []string{"multiply", "divide", "stop"},
	})
	
	// Conditional branching based on user decision
	wf.AddConditionalStep("multiply", calculator, map[string]interface{}{
		"expression": "{{.start}} * 2",
	}).WithCondition("$.check_result.decision == \"multiply\"")
	
	wf.AddConditionalStep("divide", calculator, map[string]interface{}{
		"expression": "{{.start}} / 2",
	}).WithCondition("$.check_result.decision == \"divide\"")
	
	// Define a parallel group for final operations
	wf.AddParallelGroup("final_ops", []string{"square", "cube"}, 
		workflow.WithMaxConcurrent(2),
		workflow.WithTimeout(30 * time.Second),
	)
	
	// Define steps for the parallel group
	wf.AddStep("square", calculator, map[string]interface{}{
		"expression": "{{.multiply}} * {{.multiply}}",
	}).WithCondition("exists:multiply")
	
	wf.AddStep("cube", calculator, map[string]interface{}{
		"expression": "{{.divide}} * {{.divide}} * {{.divide}}",
	}).WithCondition("exists:divide")
	
	// Create a final approval step
	wf.AddStep("approval", userInterventionTool, map[string]interface{}{
		"intervention_type": "approval",
		"title":            "Approve Final Results",
		"description":      "Final calculations complete. Square: {{.square}}, Cube: {{.cube}}. Do you approve these results?",
	})
	
	// Create a visualizer and generate a diagram
	visualizer := workflow.NewWorkflowVisualizer()
	diagram, err := visualizer.Visualize(wf, workflow.FormatMermaid)
	if err != nil {
		fmt.Printf("Error generating diagram: %v\n", err)
	} else {
		fmt.Println("Workflow Diagram:")
		fmt.Println(diagram)
	}
	
	// Execute the workflow
	fmt.Println("Executing workflow...")
	results, err := wf.Execute(context.Background(), nil)
	if err != nil {
		fmt.Printf("Workflow execution failed: %v\n", err)
		os.Exit(1)
	}
	
	fmt.Println("Workflow Results:")
	for k, v := range results {
		fmt.Printf("  %s: %v\n", k, v)
	}
}
```

## Command Line Interface

To expose these workflow capabilities through the CLI, let's add new commands:

```go
// In pkg/agent/cmd/workflow.go

/*
workflowCmd is a command for workflow operations
*/
var workflowCmd = &cobra.Command{
	Use:   "workflow",
	Short: "Manage agent workflows",
	Long:  `Create, execute, visualize, and manage agent workflows`,
}

/*
workflowRunCmd is a command to run a workflow
*/
var workflowRunCmd = &cobra.Command{
	Use:   "run [workflow_file]",
	Short: "Run a workflow",
	Long:  `Executes a workflow defined in a JSON or YAML file`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		os.Setenv("LOG_LEVEL", "debug")
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()

		// Get the workflow file
		workflowFile := args[0]
		
		// Load the workflow
		wf, err := loadWorkflowFromFile(workflowFile)
		if err != nil {
			errnie.Error(fmt.Errorf("failed to load workflow: %w", err))
			return
		}
		
		// Set up checkpointing if enabled
		checkpointDir, _ := cmd.Flags().GetString("checkpoint-dir")
		if checkpointDir != "" {
			if checkpointableWf, ok := wf.(workflow.CheckpointableWorkflow); ok {
				persistence, err := workflow.NewFileSystemPersistence(checkpointDir)
				if err != nil {
					errnie.Error(fmt.Errorf("failed to set up persistence: %w", err))
					return
				}
				
				checkpointableWf.WithPersistence(persistence)
			} else {
				errnie.Warning("Workflow does not support checkpointing")
			}
		}
		
		// Execute the workflow
		output.Title("EXECUTING WORKFLOW")
		results, err := wf.Execute(cmd.Context(), nil)
		if err != nil {
			errnie.Error(fmt.Errorf("workflow execution failed: %w", err))
			return
		}
		
		// Display results
		output.Result("Workflow completed successfully")
		output.Info("Results:")
		for k, v := range results {
			fmt.Printf("  %s: %v\n", k, v)
		}
	},
}

/*
workflowVisualizeCmd is a command to visualize a workflow
*/
var workflowVisualizeCmd = &cobra.Command{
	Use:   "visualize [workflow_file] [output_file]",
	Short: "Visualize a workflow",
	Long:  `Generates a visualization of a workflow defined in a JSON or YAML file`,
	Args:  cobra.RangeArgs(1, 2),
	Run: func(cmd *cobra.Command, args []string) {
		os.Setenv("LOG_LEVEL", "debug")
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()

		// Get the workflow file
		workflowFile := args[0]
		
		// Get the output file
		var outputFile string
		if len(args) > 1 {
			outputFile = args[1]
		}
		
		// Load the workflow
		wf, err := loadWorkflowFromFile(workflowFile)
		if err != nil {
			errnie.Error(fmt.Errorf("failed to load workflow: %w", err))
			return
		}
		
		// Get the format
		format, _ := cmd.Flags().GetString("format")
		var vizFormat workflow.Format
		switch format {
		case "mermaid":
			vizFormat = workflow.FormatMermaid
		case "dot":
			vizFormat = workflow.FormatDOT
		case "ascii":
			vizFormat = workflow.FormatASCII
		default:
			errnie.Error(fmt.Errorf("unsupported format: %s", format))
			return
		}
		
		// Create visualizer
		visualizer := workflow.NewWorkflowVisualizer()
		
		// Generate the visualization
		if outputFile == "" {
			// Print to console
			viz, err := visualizer.Visualize(wf, vizFormat)
			if err != nil {
				errnie.Error(fmt.Errorf("failed to visualize workflow: %w", err))
				return
			}
			
			fmt.Println(viz)
		} else {
			// Save to file
			err := visualizer.SaveVisualization(wf, vizFormat, outputFile)
			if err != nil {
				errnie.Error(fmt.Errorf("failed to save visualization: %w", err))
				return
			}
			
			output.Result(fmt.Sprintf("Visualization saved to %s", outputFile))
		}
	},
}

/*
workflowResumeCmd is a command to resume a workflow from a checkpoint
*/
var workflowResumeCmd = &cobra.Command{
	Use:   "resume [workflow_file] [checkpoint_id]",
	Short: "Resume a workflow from a checkpoint",
	Long:  `Resumes execution of a workflow from a saved checkpoint`,
	Args:  cobra.ExactArgs(2),
	Run: func(cmd *cobra.Command, args []string) {
		os.Setenv("LOG_LEVEL", "debug")
		os.Setenv("LOGFILE", "true")
		errnie.InitLogger()

		// Get the workflow file and checkpoint ID
		workflowFile := args[0]
		checkpointID := args[1]
		
		// Load the workflow
		wf, err := loadWorkflowFromFile(workflowFile)
		if err != nil {
			errnie.Error(fmt.Errorf("failed to load workflow: %w", err))
			return
		}
		
		// Check if workflow supports checkpointing
		checkpointableWf, ok := wf.(workflow.CheckpointableWorkflow)
		if !ok {
			errnie.Error(fmt.Errorf("workflow does not support checkpointing"))
			return
		}
		
		// Set up persistence
		checkpointDir, _ := cmd.Flags().GetString("checkpoint-dir")
		if checkpointDir == "" {
			errnie.Error(fmt.Errorf("checkpoint directory must be specified"))
			return
		}# Workflow Orchestration Enhancements

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