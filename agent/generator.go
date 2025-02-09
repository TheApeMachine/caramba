package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
)

/*
Generator is an abstraction which provides the language generation capabilities
to an Agent. It manages the interaction between the agent's configuration,
the language model provider, and the context management. The Generator handles
the streaming of responses and maintains the agent's status throughout the
generation process.
*/
type Generator struct {
	config      *Config
	provider    provider.Provider
	Ctx         *Context
	Accumulator *stream.Accumulator
	Status      AgentStatus
	Agents      map[string]*Generator
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
	return &Generator{
		config:      config,
		provider:    prvdr,
		Ctx:         NewContext(config),
		Accumulator: stream.NewAccumulator(),
		Status:      AgentStatusIdle,
		Agents:      make(map[string]*Generator),
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
	generator.Status = AgentStatusBusy

	if message.Role == provider.RoleUser {
		generator.Ctx = NewContext(generator.config)
	}

	generator.Ctx.AddMessage(message)
	generator.Ctx.Iteration++

	return generator.Accumulator.Generate(
		generator.provider.Generate(generator.Ctx.Params),
	)
}
