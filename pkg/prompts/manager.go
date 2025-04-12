package prompts

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

type ErrorPromptNotFound struct {
	ID string
}

func (err ErrorPromptNotFound) Error() string {
	return fmt.Sprintf("prompt not found: %s", err.ID)
}

type ErrorStepNotFound struct {
	ID string
}

func (err ErrorStepNotFound) Error() string {
	return fmt.Sprintf("step not found: %s", err.ID)
}

type ErrorNoStepsForPrompt struct {
	ID string
}

func (err ErrorNoStepsForPrompt) Error() string {
	return fmt.Sprintf("no steps found for prompt: %s", err.ID)
}

type ErrorInvalidPromptType struct {
	ID   string
	Type PromptType
}

func (err ErrorInvalidPromptType) Error() string {
	return fmt.Sprintf("invalid prompt type for prompt %s: %s", err.ID, err.Type)
}

type ErrorDuplicateStepOrder struct {
	PromptID string
	Order    int
}

func (err ErrorDuplicateStepOrder) Error() string {
	return fmt.Sprintf("duplicate step order %d found in prompt %s", err.Order, err.PromptID)
}

type ErrorInvalidStepOrder struct {
	PromptID string
	Order    int
}

func (err ErrorInvalidStepOrder) Error() string {
	return fmt.Sprintf("invalid step order %d in prompt %s", err.Order, err.PromptID)
}

// DefaultManager is a basic implementation of PromptManager
type DefaultManager struct {
	prompts map[string]*Prompt
	steps   map[string][]*PromptStep
	mu      sync.RWMutex
}

// NewDefaultManager creates a new DefaultManager instance
func NewDefaultManager() *DefaultManager {
	m := &DefaultManager{
		prompts: make(map[string]*Prompt),
		steps:   make(map[string][]*PromptStep),
	}

	// Add some test prompts
	m.addTestPrompts()

	return m
}

// addTestPrompts adds some test prompts to the manager
func (m *DefaultManager) addTestPrompts() {
	// Add a single-step prompt
	singlePrompt := &Prompt{
		ID:          uuid.New().String(),
		Name:        "Simple Greeting",
		Description: "A simple greeting prompt",
		Type:        SingleStepPrompt,
		Content:     "Hello, how can I help you today?",
		Version:     "1.0.0",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	m.prompts[singlePrompt.ID] = singlePrompt

	// Add a multi-step prompt
	multiPrompt := &Prompt{
		ID:          uuid.New().String(),
		Name:        "Customer Support",
		Description: "A multi-step customer support prompt",
		Type:        MultiStepPrompt,
		Content:     "This is a multi-step customer support prompt",
		Version:     "1.0.0",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	m.prompts[multiPrompt.ID] = multiPrompt

	// Add steps for the multi-step prompt
	step1 := &PromptStep{
		ID:          uuid.New().String(),
		PromptID:    multiPrompt.ID,
		Name:        "Initial Greeting",
		Description: "Initial greeting to the customer",
		Content:     "Hello, thank you for contacting our support team. How can I assist you today?",
		Order:       1,
	}
	step2 := &PromptStep{
		ID:          uuid.New().String(),
		PromptID:    multiPrompt.ID,
		Name:        "Gather Information",
		Description: "Gather information about the customer's issue",
		Content:     "Could you please provide more details about your issue?",
		Order:       2,
	}
	step3 := &PromptStep{
		ID:          uuid.New().String(),
		PromptID:    multiPrompt.ID,
		Name:        "Provide Solution",
		Description: "Provide a solution to the customer's issue",
		Content:     "Based on your description, here's what I recommend:",
		Order:       3,
	}
	step4 := &PromptStep{
		ID:          uuid.New().String(),
		PromptID:    multiPrompt.ID,
		Name:        "Closing",
		Description: "Close the conversation",
		Content:     "Is there anything else I can help you with?",
		Order:       4,
	}

	m.steps[multiPrompt.ID] = []*PromptStep{step1, step2, step3, step4}
}

// List returns all available prompts
func (m *DefaultManager) List(ctx context.Context) ([]Prompt, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	prompts := make([]Prompt, 0, len(m.prompts))
	for _, p := range m.prompts {
		prompts = append(prompts, *p)
	}

	return prompts, nil
}

// Get retrieves a prompt by ID
func (m *DefaultManager) Get(ctx context.Context, id string) (*Prompt, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	prompt, ok := m.prompts[id]
	if !ok {
		return nil, ErrorPromptNotFound{ID: id}
	}

	return prompt, nil
}

// GetSteps retrieves all steps for a multi-step prompt
func (m *DefaultManager) GetSteps(ctx context.Context, promptID string) ([]PromptStep, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Check if the prompt exists
	prompt, ok := m.prompts[promptID]
	if !ok {
		return nil, ErrorPromptNotFound{ID: promptID}
	}

	// Check if the prompt is a multi-step prompt
	if prompt.Type != MultiStepPrompt {
		return nil, ErrorInvalidPromptType{ID: promptID, Type: prompt.Type}
	}

	// Get the steps
	steps, ok := m.steps[promptID]
	if !ok {
		return nil, ErrorNoStepsForPrompt{ID: promptID}
	}

	// Convert to slice of PromptStep
	result := make([]PromptStep, len(steps))
	for i, step := range steps {
		result[i] = *step
	}

	return result, nil
}

// Create creates a new prompt
func (m *DefaultManager) Create(ctx context.Context, prompt Prompt) (*Prompt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Generate a new ID if not provided
	if prompt.ID == "" {
		prompt.ID = uuid.New().String()
	}

	// Set timestamps
	now := time.Now()
	prompt.CreatedAt = now
	prompt.UpdatedAt = now

	// Store the prompt
	m.prompts[prompt.ID] = &prompt

	// Initialize steps for multi-step prompts
	if prompt.Type == MultiStepPrompt {
		m.steps[prompt.ID] = make([]*PromptStep, 0)
	}

	return &prompt, nil
}

// Update updates an existing prompt
func (m *DefaultManager) Update(ctx context.Context, prompt Prompt) (*Prompt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if the prompt exists
	existingPrompt, ok := m.prompts[prompt.ID]

	if !ok {
		return nil, ErrorPromptNotFound{ID: prompt.ID}
	}

	// Update the prompt
	prompt.CreatedAt = existingPrompt.CreatedAt
	prompt.UpdatedAt = time.Now()
	m.prompts[prompt.ID] = &prompt

	return &prompt, nil
}

// Delete deletes a prompt
func (m *DefaultManager) Delete(ctx context.Context, id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if the prompt exists
	_, ok := m.prompts[id]
	if !ok {
		return ErrorPromptNotFound{ID: id}
	}

	// Delete the prompt
	delete(m.prompts, id)

	// Delete the steps if it's a multi-step prompt
	delete(m.steps, id)

	return nil
}

// CreateStep creates a new step for a multi-step prompt
func (m *DefaultManager) CreateStep(ctx context.Context, step PromptStep) (*PromptStep, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if the prompt exists
	prompt, ok := m.prompts[step.PromptID]
	if !ok {
		return nil, ErrorPromptNotFound{ID: step.PromptID}
	}

	// Check if the prompt is a multi-step prompt
	if prompt.Type != MultiStepPrompt {
		return nil, ErrorInvalidPromptType{ID: step.PromptID, Type: prompt.Type}
	}

	// Generate a new ID if not provided
	if step.ID == "" {
		step.ID = uuid.New().String()
	}

	// Get the existing steps
	steps, ok := m.steps[step.PromptID]
	if !ok {
		steps = make([]*PromptStep, 0)
	}

	// Validate step order
	if step.Order <= 0 {
		return nil, ErrorInvalidStepOrder{PromptID: step.PromptID, Order: step.Order}
	}

	// Check for duplicate order
	for _, s := range steps {
		if s.Order == step.Order {
			return nil, ErrorDuplicateStepOrder{PromptID: step.PromptID, Order: step.Order}
		}
	}

	// Add the new step
	steps = append(steps, &step)
	m.steps[step.PromptID] = steps

	return &step, nil
}

// UpdateStep updates an existing step
func (m *DefaultManager) UpdateStep(ctx context.Context, step PromptStep) (*PromptStep, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if the prompt exists
	_, ok := m.prompts[step.PromptID]
	if !ok {
		return nil, ErrorPromptNotFound{ID: step.PromptID}
	}

	// Get the existing steps
	steps, ok := m.steps[step.PromptID]
	if !ok {
		return nil, ErrorNoStepsForPrompt{ID: step.PromptID}
	}

	// Find and update the step
	found := false
	for i, s := range steps {
		if s.ID == step.ID {
			steps[i] = &step
			found = true
			break
		}
	}

	if !found {
		return nil, ErrorStepNotFound{ID: step.ID}
	}

	return &step, nil
}

// DeleteStep deletes a step
func (m *DefaultManager) DeleteStep(ctx context.Context, stepID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Find the step
	var promptID string
	var stepIndex int
	found := false

	for pid, steps := range m.steps {
		for i, step := range steps {
			if step.ID == stepID {
				promptID = pid
				stepIndex = i
				found = true
				break
			}
		}
		if found {
			break
		}
	}

	if !found {
		return ErrorStepNotFound{ID: stepID}
	}

	// Remove the step
	steps := m.steps[promptID]
	m.steps[promptID] = append(steps[:stepIndex], steps[stepIndex+1:]...)

	return nil
}
