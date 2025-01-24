package ai

import (
	"context"

	"github.com/theapemachine/caramba/provider"
)

// Process represents a structured workflow for handling specific types of tasks
type Process interface {
	// Initialize sets up any necessary resources or state
	Initialize(ctx context.Context) error

	// GeneratePrompt creates the next prompt based on current state
	GeneratePrompt(role string, state interface{}) (*provider.Message, error)

	// ValidateResponse checks if a response meets the process requirements
	ValidateResponse(response interface{}) (interface{}, error)

	// UpdateState updates the process state with new information
	UpdateState(agentID string, response interface{}) error

	// IsComplete checks if the process has reached its goal
	IsComplete() bool
}
