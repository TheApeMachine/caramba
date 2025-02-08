package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tweaker"
	"github.com/theapemachine/caramba/utils"
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
	config      *Config
	provider    provider.Provider
	ctx         *Context
	accumulator *stream.Accumulator
	status      AgentStatus
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
		ctx:         NewContext(config),
		accumulator: stream.NewAccumulator(),
		status:      AgentStatusIdle,
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
	out := make(chan *provider.Event)

	go func() {
		defer close(out)

		generator.status = AgentStatusBusy
		generator.ctx.AddUserPrompt(message)

		generator.accumulator.After(func(str string) {
			generator.after()
		})

		for {
			generator.ctx.iteration++

			for event := range generator.accumulator.Generate(
				generator.provider.Generate(generator.ctx.params),
			) {
				out <- event
			}

			for _, msg := range generator.ctx.config.Thread.Messages {
				errnie.Log(msg.Content)
			}
		}
	}()

	return out
}

/*
after performs post-generation processing steps. It handles tool calls and
updates the conversation thread with the generated response.
*/
func (generator *Generator) after() {
	generator.toolcalls()

	generator.ctx.config.Thread.AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			tweaker.GetIteration(
				generator.ctx.config.Name,
				generator.ctx.config.Role,
				generator.ctx.iteration,
				generator.accumulator.String(),
			),
		),
	)
}

/*
toolcalls processes any tool invocations found in the generated response.
It extracts JSON blocks from the response and executes the corresponding
tool calls.
*/
func (generator *Generator) toolcalls() {
	blocks := utils.ExtractJSONBlocks(generator.accumulator.String())
	for _, block := range blocks {
		if toolname, ok := block["tool"].(string); ok {
			if args, ok := block["args"].(map[string]any); ok {
				generator.toolcall(toolname, args)
			}
		}
	}
}

/*
toolcall executes a specific tool with the provided arguments and updates
the generator's status based on the tool's response.

Parameters:

	toolname: The name of the tool to execute
	args: A map of arguments to pass to the tool

Returns:

	string: The result of the tool execution
*/
func (generator *Generator) toolcall(toolname string, args map[string]any) {
	generator.accumulator.Append(
		generator.updateStatus(
			generator.config.Toolset.Use(toolname, args),
		),
	)
}

/*
updateStatus updates the generator's status based on the provided string
and returns the same string.

Parameters:

	str: The string to evaluate for status update

Returns:

	string: The input string, unmodified
*/
func (generator *Generator) updateStatus(str string) string {
	switch str {
	case "break":
		generator.status = AgentStatusIdle
	default:
		generator.status = AgentStatusBusy
	}

	return str
}
