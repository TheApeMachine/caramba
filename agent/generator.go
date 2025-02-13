package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/types"
	"github.com/theapemachine/errnie"
)

/*
Generator is an abstraction which provides the language generation capabilities
to an Agent. It manages the interaction between the agent's configuration,
the language model provider, and the context management. The Generator handles
the streaming of responses and maintains the agent's status throughout the
generation process.
*/
type Generator struct {
	config      types.Config
	provider    provider.Provider
	ctx         types.Context
	accumulator *stream.Accumulator
	status      types.AgentStatus
	agents      map[string]types.Generator
}

/*
NewGenerator creates and returns a new Generator instance.

Parameters:

	config: The agent configuration that defines behavior and settings
	prvdr: The language model provider that will handle the actual generation

Returns:

	*Generator: A new Generator instance initialized with the provided config and provider
*/
func NewGenerator(
	config *Config, prvdr provider.Provider,
) *Generator {
	errnie.Debug("new generator", "config", config.Name, "role", config.Role)

	return &Generator{
		config:      config,
		provider:    prvdr,
		ctx:         NewContext(config),
		accumulator: stream.NewAccumulator(),
		status:      types.AgentStatusIdle,
		agents:      make(map[string]types.Generator),
	}
}

/*
Generate processes a user message and generates a response using the configured
language model provider. It handles the streaming of the response and manages
the generator's status throughout the process.

Parameters:

	message: The user message to process and generate a response for

Returns:

	<-chan *provider.Event: A channel that streams the generated response events
*/
func (generator *Generator) Generate(
	message *provider.Message,
) <-chan *provider.Event {
	generator.status = types.AgentStatusBusy
	generator.accumulator.Clear()

	if message.Role == provider.RoleUser {
		generator.ctx = NewContext(generator.config)
	}

	generator.ctx.AddMessage(message)
	generator.ctx.SetIteration(generator.ctx.Iteration() + 1)

	return generator.accumulator.Generate(
		generator.provider.Generate(generator.ctx.Params()),
	)
}

func (generator *Generator) Status() types.AgentStatus {
	return generator.status
}

func (generator *Generator) SetStatus(status types.AgentStatus) {
	generator.status = status
}

func (generator *Generator) Accumulator() *stream.Accumulator {
	return generator.accumulator
}

func (generator *Generator) Ctx() types.Context {
	return generator.ctx
}

func (generator *Generator) Agents() map[string]types.Generator {
	return generator.agents
}
