package ai

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/errnie"
	"github.com/theapemachine/qpool"
)

type Orchestrator struct {
	ctx             context.Context
	Agents          map[string][]*Agent
	consensus       *drknow.ConsensusSpace
	pool            *qpool.Q
	taskConfig      *TaskConfig
	pythonContainer *tools.PythonContainer
}

type TaskConfig struct {
	InitialPrompt   string
	ValidationRules []ValidationRule
	ConsensusConfig drknow.ConsensusConfig
	RequiredAgents  []string
	MaxIterations   int
}

type ValidationRule struct {
	Name      string
	Validator func(interface{}) (interface{}, error)
}

func NewOrchestrator(ctx context.Context, config *TaskConfig) *Orchestrator {
	o := &Orchestrator{
		ctx:        ctx,
		Agents:     make(map[string][]*Agent),
		taskConfig: config,
		pool: qpool.NewQ(ctx, 4, 8, &qpool.Config{
			SchedulingTimeout: time.Second * 60,
		}),
		pythonContainer: tools.NewPythonContainer(),
		consensus:       drknow.NewConsensusSpace("analysis-space", config.ConsensusConfig),
	}

	if err := o.pythonContainer.Initialize(); err != nil {
		errnie.Error(err)
	}

	return o
}

func (o *Orchestrator) GeneratePrompt(agentRole string, context interface{}) (*provider.Message, error) {
	promptRequest := map[string]interface{}{
		"role":        agentRole,
		"context":     context,
		"task":        o.taskConfig.InitialPrompt,
		"constraints": o.taskConfig.ValidationRules,
	}

	return &provider.Message{
		Role:    "user",
		Content: fmt.Sprintf("%v", promptRequest),
	}, nil
}

func (o *Orchestrator) ValidateResponse(response interface{}) (interface{}, error) {
	for _, rule := range o.taskConfig.ValidationRules {
		var err error
		response, err = rule.Validator(response)
		if err != nil {
			return nil, fmt.Errorf("validation failed for rule %s: %w", rule.Name, err)
		}
	}
	return response, nil
}

func (o *Orchestrator) extractAndExecutePython(response string) (string, error) {
	// Look for Python code blocks
	pythonBlocks := extractCodeBlocks(response, "python")
	if len(pythonBlocks) == 0 {
		return response, nil
	}

	var results []string
	for _, code := range pythonBlocks {
		result, err := o.pythonContainer.ExecutePython(o.ctx, code)
		if err != nil {
			return response, err
		}
		results = append(results, result)
	}

	// Append results to response
	return fmt.Sprintf("%s\n\nPython Execution Results:\n%s",
		response, strings.Join(results, "\n")), nil
}

func extractCodeBlocks(text, language string) []string {
	var blocks []string
	marker := "```" + language
	parts := strings.Split(text, marker)

	for i := 1; i < len(parts); i += 2 {
		if i >= len(parts) {
			break
		}
		code := strings.Split(parts[i], "```")[0]
		blocks = append(blocks, strings.TrimSpace(code))
	}

	return blocks
}

func (o *Orchestrator) ProcessAgentResponse(agentID string, response interface{}) error {
	// Execute any Python code in the response
	if strResponse, ok := response.(string); ok {
		enrichedResponse, err := o.extractAndExecutePython(strResponse)
		if err != nil {
			return err
		}
		response = enrichedResponse
	}

	// Continue with existing validation and consensus logic
	validated, err := o.ValidateResponse(response)
	if err != nil {
		return err
	}

	perspective := drknow.Perspective{
		ID:         agentID,
		Owner:      agentID,
		Content:    validated,
		Confidence: o.calculateConfidence(validated),
		Method:     "dynamic-analysis",
		Timestamp:  time.Now(),
	}

	o.consensus.AddPerspective(perspective)
	return nil
}

func (o *Orchestrator) RunProcess(process Process) error {
	if err := process.Initialize(o.ctx); err != nil {
		return fmt.Errorf("failed to initialize process: %w", err)
	}

	// Set up dependencies
	o.consensus.AddDependency("reasoner", []string{"prompt_engineer"})
	o.consensus.AddDependency("challenger", []string{"reasoner"})
	o.consensus.AddDependency("solver", []string{"challenger", "reasoner"})

	errnie.Info("Starting to schedule agents")

	// Create a wait group for all agents
	var wg sync.WaitGroup
	wg.Add(len(o.taskConfig.RequiredAgents))

	// Channel to collect errors from agents
	errChan := make(chan error, len(o.taskConfig.RequiredAgents))

	// Track completed agents
	completedAgents := make(map[string]bool)
	var completedMutex sync.Mutex

	// Helper function to check if dependencies are met
	checkDependencies := func(role string) bool {
		deps := o.consensus.GetDependencies(role)
		completedMutex.Lock()
		defer completedMutex.Unlock()

		for _, dep := range deps {
			if !completedAgents[dep] {
				return false
			}
		}
		return true
	}

	// Helper function to mark agent as completed
	markCompleted := func(role string) {
		completedMutex.Lock()
		completedAgents[role] = true
		completedMutex.Unlock()
	}

	// Schedule agents with dependency checking
	for _, agentRole := range o.taskConfig.RequiredAgents {
		go func(role string) {
			// Wait for dependencies
			for !checkDependencies(role) {
				time.Sleep(100 * time.Millisecond)
			}

			errnie.Info("Scheduling agent: %s", role)
			if err := o.scheduleAgent(role, process); err != nil {
				errChan <- fmt.Errorf("failed to schedule agent %s: %w", role, err)
				return
			}

			markCompleted(role)
		}(agentRole)
	}

	// Add logging for consensus events
	o.consensus.OnNewPerspective = func(p drknow.Perspective) {
		errnie.Info("New perspective added from %s with confidence %f", p.Owner, p.Confidence)
		content := ""
		if str, ok := p.Content.(string); ok {
			content = str
		}
		errnie.Info("Content: %s", content)
	}

	o.consensus.OnCollapse = func(result interface{}) {
		errnie.Info("Consensus reached with result: %v", result)
		wg.Done()
	}

	// Add timeout for the entire process
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case err := <-errChan:
		return fmt.Errorf("agent failed: %w", err)
	case <-time.After(5 * time.Minute):
		return fmt.Errorf("process timed out waiting for consensus")
	}
}

func (o *Orchestrator) scheduleAgent(role string, process Process) error {
	resultChan := o.pool.Schedule(role, func() (any, error) {
		errnie.Info("Agent %s is starting", role)

		// Generate contextual prompt
		prompt, err := process.GeneratePrompt(role, nil)
		if err != nil {
			errnie.Info("Failed to generate prompt for %s: %v", role, err)
			return nil, fmt.Errorf("failed to generate prompt: %w", err)
		}

		// Get agent for role
		agent := o.Agents[role][0]

		// Generate response
		responseChan := agent.Generate(o.ctx, prompt)

		// Accumulate response
		accumulator := stream.NewAccumulator()
		for event := range accumulator.Generate(o.ctx, responseChan) {
			if data, ok := event.Data().(map[string]interface{}); ok {
				if text, ok := data["text"].(string); ok {
					errnie.Info("Agent %s generated: %s", role, text)
				}
			}
		}

		// Get the complete response
		response := accumulator.Compile()
		data := response.Data().(map[string]interface{})
		text, ok := data["text"].(string)
		if !ok || text == "" {
			return nil, fmt.Errorf("empty response from agent %s", role)
		}

		// Validate response
		validated, err := process.ValidateResponse(text)
		if err != nil {
			errnie.Info("Validation failed for %s: %v", role, err)
			return nil, fmt.Errorf("validation failed: %w", err)
		}

		// Update process state
		if err := process.UpdateState(role, validated); err != nil {
			errnie.Info("Failed to update state for %s: %v", role, err)
			return nil, fmt.Errorf("failed to update state: %w", err)
		}

		// Add to consensus space with higher confidence for later agents
		confidence := 0.8
		if role == "solver" {
			confidence = 0.9 // Give solver higher weight
		}

		perspective := drknow.Perspective{
			ID:         role,
			Owner:      role,
			Content:    validated,
			Confidence: confidence,
			Method:     "analysis",
			Timestamp:  time.Now(),
		}

		o.consensus.AddPerspective(perspective)
		errnie.Info("Added perspective for %s with content: %v", role, validated)

		return validated, nil
	})

	// Handle the result channel
	go func() {
		for result := range resultChan {
			if result.Error != nil {
				errnie.Error(fmt.Errorf("agent %s failed: %w", role, result.Error))
				continue
			}
			errnie.Info("Agent %s completed successfully", role)
		}
	}()

	return nil
}

func (o *Orchestrator) calculateConfidence(response interface{}) float64 {
	// Simple implementation - can be enhanced based on your needs
	if response == nil {
		return 0.0
	}
	return 0.8 // Default confidence level
}
