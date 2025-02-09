package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tweaker"
)

/*
Context manages the conversation state and generation parameters for an agent.
It maintains the configuration, generation parameters, and keeps track of
the current iteration in the conversation flow. The Context is responsible
for managing the conversation history and ensuring proper message formatting.
*/
type Context struct {
	Config    *Config                       // Agent configuration
	Params    *provider.LLMGenerationParams // Parameters for language model generation
	Iteration int                           // Current conversation iteration count
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
func NewContext(config *Config) *Context {
	return &Context{
		Config:    config,
		Params:    provider.NewGenerationParams(config.Thread),
		Iteration: 0,
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
		ctx.Config.Thread.AddMessage(
			provider.NewMessage(
				provider.RoleUser,
				tweaker.GetUserPrompt(message.Content),
			),
		)

		ctx.Config.Thread.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				tweaker.GetContext(),
			),
		)

		return
	}

	ctx.Config.Thread.AddMessage(
		provider.NewMessage(message.Role, message.Content),
	)
}
