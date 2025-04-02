package ai

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/errnie"
)

type AgentValidationError struct {
	err   error
	scope string
}

func NewAgentValidationError(scope string, err error) *AgentValidationError {
	return &AgentValidationError{
		err:   errnie.Error(fmt.Errorf("%s - %w", scope, err)),
		scope: scope,
	}
}

func (e *AgentValidationError) Error() string {
	return e.err.Error()
}

func (e *AgentValidationError) Unwrap() error {
	return e.err
}

type AgentContextValidationError struct {
	err   error
	scope string
}

func NewAgentContextValidationError(scope string, err error) *AgentContextValidationError {
	return &AgentContextValidationError{
		err:   errnie.Error(fmt.Errorf("%s - %w", scope, err)),
		scope: scope,
	}
}

func (e *AgentContextValidationError) Error() string {
	return e.err.Error()
}

func (e *AgentContextValidationError) Unwrap() error {
	return e.err
}
