package drknow

import (
	"context"
	"strings"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

/*
Context is a wrapper that manages the current conversation thread and response generation.
It handles structural formatting, message accumulation, and tool call tracking while
ensuring proper alignment of nested content.

Context maintains the conversation state, including the identity of the agent,
a scratchpad for accumulating responses, and a record of tool calls made during
the conversation.
*/
type Context struct {
	Identity  *Identity
	Toolcalls []*provider.Event
	indent    int
}

/*
NewContext creates a new Context instance with the specified Identity.
It initializes an empty scratchpad for accumulating assistant responses
and sets up the basic formatting configuration.

Parameters:
  - identity: The Identity configuration for the agent using this context

Returns:
  - A new Context instance ready for message management
*/
func NewContext(identity *Identity) *Context {
	return &Context{
		Identity: identity,
		indent:   0,
	}
}

func QuickContext(
	system string,
	additions ...string,
) *Context {
	v := viper.GetViper()
	steerings := "prompts.templates.steering."

	steering := make([]string, 0)
	for _, addition := range additions {
		steering = append(
			steering,
			strings.TrimSpace(v.GetString(steerings+addition)),
		)
	}

	return NewContext(
		NewIdentity(
			context.Background(),
			"reasoner",
			utils.JoinWith(
				"\n\n",
				system,
				strings.Join(steering, "\n"),
			),
		),
	)
}

/*
Compile prepares the context for a new generation by resetting and rebuilding
the conversation thread. This is called at the start of each generation to ensure
that any self-optimizing changes to the system prompt are included.

The method combines the system prompt, the input message, and the current scratchpad
into a properly formatted conversation thread.

Parameters:
  - msg: The input message to be added to the conversation thread

Returns:
  - Generation parameters containing the compiled conversation thread
*/
func (ctx *Context) Compile(cycle int, maxIterations int) *provider.LLMGenerationParams {
	return ctx.Identity.Params
}

/*
String compiles the current context and returns it as a string.
*/
func (ctx *Context) String(includeSystem bool) string {
	builder := strings.Builder{}

	for _, message := range ctx.Identity.Params.Thread.Messages {
		if !includeSystem && message.Role == provider.RoleSystem {
			continue
		}

		builder.WriteString(utils.JoinWith("\n",
			string(message.Role),
			strings.TrimSpace(message.Content)+"\n\n",
		))
	}

	return strings.TrimSpace(builder.String())
}

/*
Reset the context to start fresh
*/
func (ctx *Context) Reset() {
	ctx.Identity.Params.Thread.Messages = make([]*provider.Message, 0)
	ctx.Identity.Params.Thread.Messages = append(
		ctx.Identity.Params.Thread.Messages,
		provider.NewMessage(
			provider.RoleSystem,
			strings.TrimSpace(ctx.Identity.System),
		),
	)
	ctx.Toolcalls = make([]*provider.Event, 0)
}

/*
AddMessage adds a new message to the context.
*/
func (ctx *Context) AddMessage(msg *provider.Message) {
	ctx.Identity.Params.Thread.AddMessage(msg)
}

/*
LastMessage ...
*/
func (ctx *Context) LastMessage() *provider.Message {
	if len(ctx.Identity.Params.Thread.Messages) == 0 {
		return nil
	}
	return ctx.Identity.Params.Thread.Messages[len(ctx.Identity.Params.Thread.Messages)-1]
}
