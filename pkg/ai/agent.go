package ai

import (
	"github.com/theapemachine/caramba/pkg/errnie"
)

/*
Agent represents an entity that can process messages, interact with tools,
and produce responses. It implements io.ReadWriteCloser to enable composable
pipelines for complex workflows.
*/
type Agent struct {
	Context *Context `json:"context"`
}

/*
NewAgent creates a new agent with initialized components.
*/
func NewAgent() *Agent {
	errnie.Debug("Creating new agent", "package", "agent")

	return &Agent{
		Context: NewContext(),
	}
}
