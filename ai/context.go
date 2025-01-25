package ai

import (
	"errors"
	"os"
	"strings"

	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
	"github.com/theapemachine/errnie"
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
	identity   *Identity
	Scratchpad *provider.Message
	Toolcalls  []*provider.Event
	indent     int
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
		identity:   identity,
		Scratchpad: provider.NewMessage(provider.RoleAssistant, ""),
		indent:     0,
	}
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
func (ctx *Context) Compile(msg *provider.Message, cycle int, maxIterations int) *provider.LLMGenerationParams {
	if ctx.identity == nil {
		errnie.Error(errors.New("identity is nil"))
		os.Exit(1)
	}

	if ctx.identity.Params == nil {
		errnie.Error(errors.New("params are nil"))
		os.Exit(1)
	}

	if ctx.identity.Params.Thread == nil {
		errnie.Error(errors.New("thread is nil"))
		os.Exit(1)
	}

	ctx.identity.Params.Thread.Reset()

	ctx.identity.Params.Thread.AddMessage(
		provider.NewMessage(provider.RoleSystem, ctx.identity.System),
	).AddMessage(msg)

	// Make sure we don't add empty messages to the thread, it pisses Claude off.
	if ctx.Scratchpad != nil && ctx.Scratchpad.Content != "" {
		ctx.identity.Params.Thread.AddMessage(ctx.Scratchpad)
	}

	return ctx.identity.Params
}

/*
String compiles the current context and returns it as a string.
*/
func (ctx *Context) String() string {
	builder := strings.Builder{}

	builder.WriteString(utils.JoinWith("\n",
		"SYSTEM PROMPT:",
		ctx.identity.System,
	))
	builder.WriteString("\n\n")

	for _, message := range ctx.identity.Params.Thread.Messages {
		builder.WriteString(utils.JoinWith("\n",
			string(message.Role),
			message.Content,
		))
		builder.WriteString("\n\n")
	}

	return builder.String()
}

/*
Append adds a new event to the context based on its type.
It handles different types of events appropriately:
  - Tool calls are collected in the Toolcalls slice
  - Text chunks are added to the Scratchpad
  - Errors are recorded in the Scratchpad

Parameters:
  - event: The provider.Event to be processed and added to the context
*/
func (ctx *Context) Append(event provider.Event) {
	switch event.Type() {
	case "tool_call":
	switch event.Type() {
	case "tool_call":
		ctx.Toolcalls = append(ctx.Toolcalls, &event)
	case "chunk":
		if data, ok := event.Data().(map[string]interface{}); ok {
			if text, ok := data["text"].(string); ok {
				ctx.Scratchpad.Append(text)
			}
		}
	case "error":
		if data, ok := event.Data().(map[string]interface{}); ok {
			if text, ok := data["text"].(string); ok {
				ctx.Scratchpad.Append(text)
			}
		}
	case "chunk":
		if data, ok := event.Data().(map[string]interface{}); ok {
			if text, ok := data["text"].(string); ok {
				ctx.Scratchpad.Append(text)
			}
		}
	case "error":
		if data, ok := event.Data().(map[string]interface{}); ok {
			if text, ok := data["text"].(string); ok {
				ctx.Scratchpad.Append(text)
			}
		}
	}
}
