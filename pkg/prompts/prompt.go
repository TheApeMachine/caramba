package prompts

import (
	"context"
	"time"
)

// PromptType represents the type of prompt
type PromptType string

const (
	// SingleStepPrompt is a simple one-step prompt
	SingleStepPrompt PromptType = "single"
	// MultiStepPrompt is a prompt that requires multiple steps
	MultiStepPrompt PromptType = "multi"
)

// Prompt represents a prompt that can be used for various purposes
type Prompt struct {
	ID          string     `json:"id"`
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Type        PromptType `json:"type"`
	Content     string     `json:"content"`
	Version     string     `json:"version"`
	CreatedAt   time.Time  `json:"createdAt"`
	UpdatedAt   time.Time  `json:"updatedAt"`
	Metadata    any        `json:"metadata,omitempty"`
}

// PromptStep represents a step in a multi-step prompt
type PromptStep struct {
	ID          string `json:"id"`
	PromptID    string `json:"promptId"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Content     string `json:"content"`
	Order       int    `json:"order"`
	Metadata    any    `json:"metadata,omitempty"`
}

// PromptManager defines the interface for managing prompts
type PromptManager interface {
	// List returns all available prompts
	List(ctx context.Context) ([]Prompt, error)

	// Get retrieves a prompt by ID
	Get(ctx context.Context, id string) (*Prompt, error)

	// GetSteps retrieves all steps for a multi-step prompt
	GetSteps(ctx context.Context, promptID string) ([]PromptStep, error)

	// Create creates a new prompt
	Create(ctx context.Context, prompt Prompt) (*Prompt, error)

	// Update updates an existing prompt
	Update(ctx context.Context, prompt Prompt) (*Prompt, error)

	// Delete deletes a prompt
	Delete(ctx context.Context, id string) error

	// CreateStep creates a new step for a multi-step prompt
	CreateStep(ctx context.Context, step PromptStep) (*PromptStep, error)

	// UpdateStep updates an existing step
	UpdateStep(ctx context.Context, step PromptStep) (*PromptStep, error)

	// DeleteStep deletes a step
	DeleteStep(ctx context.Context, stepID string) error
}
