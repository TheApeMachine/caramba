package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
)

type ExecutorStatus uint

const (
    
	ExecutorStatusIdle = iota
	ExecutorStatusBusy
)

type Executor struct {
	config      *Config
	generator   *Generator
	accumulator *stream.Accumulator
	status      AgentStatus
}

func NewExecutor(
	config *Config, generator *Generator,
) *Executor {
	return &Executor{
		config:      config,
		generator:   generator,
		accumulator: stream.NewAccumulator(),
	}
}

func (executor *Executor) Generate(message *provider.Message) <-chan *provider.Event {
	executor.status = AgentStatusBusy
	executor.accumulator.Clear()
	executor.generator.Generate(message)

	executor.accumulator.After(func(str string) {
		executor.status = AgentStatusIdle
	})

	return executor.accumulator.Generate(
		executor.generator.Generate(message),
	)
}
