package ai

import (
	"context"

	"github.com/theapemachine/caramba/provider"
)

// AgentInterface defines the interface for AI agents in the system.
type AgentInterface interface {
	Initialize()
	AddTools(tools ...provider.Tool)
	AddProcess(process provider.Process)
	GetRole() string
	Generate(ctx context.Context, message *provider.Message) <-chan provider.Event
}
