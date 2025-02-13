package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tweaker"
	"github.com/theapemachine/caramba/types"
)

/*
Context manages the conversation state and generation parameters for an agent.
It maintains the configuration, generation parameters, and keeps track of
the current iteration in the conversation flow. The Context is responsible
for managing the conversation history and ensuring proper message formatting.
*/
type Context struct {
	config    types.Config                  // Agent configuration
	params    *provider.LLMGenerationParams // Parameters for language model generation
	iteration int                           // Current conversation iteration count
}

/*
NewContext creates and returns a new Context instance with the provided
configuration. It initializes the generation parameters and sets the
iteration counter to zero.

Parameters:

	config: The agent configuration to use for this context

Returns:

	*Context: A new Context instance initialized with the provided configuration
*/
func NewContext(config types.Config) types.Context {
	return &Context{
		config:    config,
		params:    provider.NewGenerationParams(config.Thread()),
		iteration: 0,
	}
}

/*
AddMessage adds a message to the conversation thread and prepares
the context for the next generation step. It formats the message
according to the configured tweaks and adds both the message and
an assistant context message to the thread.

Parameters:

	message: The user message to add to the conversation thread
*/
func (ctx *Context) AddMessage(
	message *provider.Message,
) {
	if message.Content == "" {
		return
	}

	if message.Role == provider.RoleUser {
		ctx.config.Thread().AddMessage(
			provider.NewMessage(
				provider.RoleUser,
				tweaker.GetUserPrompt(message.Content),
			),
		)

		return
	}

	ctx.config.Thread().AddMessage(
		provider.NewMessage(message.Role, message.Content),
	)
}

func (ctx *Context) Params() *provider.LLMGenerationParams {
	return ctx.params
}

func (ctx *Context) Iteration() int {
	return ctx.iteration
}

func (ctx *Context) SetIteration(iteration int) {
	ctx.iteration = iteration
}

func (ctx *Context) Config() types.Config {
	return ctx.config
}

func (ctx *Context) Thread() *provider.Thread {
	return ctx.config.Thread()
}
