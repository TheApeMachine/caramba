package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
)

/*
ExecutorStatus represents the current operational state of an executor.
It is used to track whether an executor is currently processing a request
or is available for new tasks.
*/
type ExecutorStatus uint

/*
Executor status constants define the possible states an executor can be in:

	ExecutorStatusIdle: The executor is not currently processing any requests
	ExecutorStatusBusy: The executor is actively processing a request
*/
const (
	ExecutorStatusIdle = iota // Executor is available for new tasks
	ExecutorStatusBusy        // Executor is currently processing a task
)

/*
Executor manages the execution flow of agent operations. It coordinates
between the configuration, generator, and accumulator components to process
requests and manage the agent's state during generation.
*/
type Executor struct {
	config      *Config             // Agent configuration
	generator   *Generator          // Language generation component
	accumulator *stream.Accumulator // Stream accumulator for responses
	status      AgentStatus         // Current executor status
}

/*
NewExecutor creates and returns a new Executor instance with the provided
configuration and generator components.

Parameters:

	config: The agent configuration to use
	generator: The generator component for language generation

Returns:

	*Executor: A new Executor instance initialized with the provided components
*/
func NewExecutor(
	config *Config, generator *Generator,
) *Executor {
	return &Executor{
		config:      config,
		generator:   generator,
		accumulator: stream.NewAccumulator(),
	}
}

/*
Generate processes a message through the generator and returns a channel
of events. It manages the executor's status during generation and ensures
proper cleanup after generation is complete.

Parameters:

	message: The message to process

Returns:

	<-chan *provider.Event: A channel that streams the generated response events
*/
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
