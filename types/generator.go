package types

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
)

type AgentStatus uint

const (
	AgentStatusIdle AgentStatus = iota
	AgentStatusBusy
	AgentStatusDone
	AgentStatusError
)

type Generator interface {
	Generate(*provider.Message) <-chan *provider.Event
	Status() AgentStatus
	SetStatus(AgentStatus)
	Accumulator() *stream.Accumulator
	Ctx() Context
	Agents() map[string]Generator
}

type Context interface {
	Config() Config
	Thread() *provider.Thread
	Iteration() int
	SetIteration(int)
	AddMessage(*provider.Message)
	Params() *provider.LLMGenerationParams
}

type Config interface {
	Name() string
	Role() string
	SystemPrompt() string
	Thread() *provider.Thread
	Temperature() float32
}
