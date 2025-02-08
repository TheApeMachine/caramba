package drknow

import (
	"context"
	"strconv"
	"strings"

	"github.com/spf13/viper"
	"github.com/theapemachine/caramba/markymark"
	"github.com/theapemachine/caramba/provider"
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
	Context    context.Context
	Cancel     context.CancelFunc
	Identity   *Identity
	system     *provider.Message
	user       *provider.Message
	iterations [][]*provider.Message
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
func NewContext(identity *Identity, ctx context.Context, cancel context.CancelFunc) *Context {
	return &Context{
		Context:    ctx,
		Cancel:     cancel,
		Identity:   identity,
		system:     provider.NewMessage(provider.RoleSystem, identity.System),
		iterations: make([][]*provider.Message, 0),
	}
}

func QuickContext(
	system string,
	additions ...string,
) *Context {
	ctx, cancel := context.WithCancel(context.Background())

	return NewContext(
		NewIdentity(
			context.Background(),
			"reasoner",
			system,
		),
		ctx,
		cancel,
	)
}

/*
writeLog performs specialized formatting to make sure the logs are easily readable.
*/
func (ctx *Context) writeLog(params *provider.LLMGenerationParams) {
	if params == nil {
		return
	}

	for _, message := range params.Thread.Messages {
		for _, line := range strings.Split(message.Content, "\n") {
			errnie.Log("%s\n", line)
		}
	}
}

/*
Compile prepares the context for a new generation by resetting and rebuilding
the conversation thread. This is called at the start of each generation to ensure
that any self-optimizing changes to the system prompt are included.

Returns:
  - Generation parameters containing the compiled conversation thread
*/
func (ctx *Context) Compile() *provider.LLMGenerationParams {
	errnie.Debug("compiling context", "role", ctx.Identity.Role)

	params := provider.NewGenerationParams(provider.NewThread())
	params.Thread.AddMessage(provider.NewMessage(provider.RoleSystem, ctx.Identity.System))
	params.Thread.AddMessage(ctx.user)

	if len(ctx.iterations) == 0 {
		ctx.writeLog(params)
		return params
	}

	var out strings.Builder

	for idx, iteration := range ctx.iterations {
		var iterout strings.Builder
		for _, message := range iteration {
			iterout.WriteString(message.Content)
		}

		tmpl := viper.GetViper().GetString("prompts.templates.tasks.iteration")
		tmpl = strings.ReplaceAll(tmpl, "<{iteration}>", strconv.Itoa(idx+1))
		tmpl = strings.ReplaceAll(tmpl, "<{response}>", iterout.String())

		out.WriteString(tmpl)
	}

	tmpl := viper.GetViper().GetString("prompts.templates.tasks.context")
	tmpl = strings.ReplaceAll(tmpl, "<{context}>", out.String())

	params.Thread.AddMessage(provider.NewMessage(provider.RoleAssistant, tmpl))

	ctx.writeLog(params)
	ctx.Reset()
	return params
}

/*
String compiles the current context and returns it as a string.
*/
func (ctx *Context) String(includeSystem bool) string {
	params := ctx.Compile()

	builder := strings.Builder{}

	for _, message := range params.Thread.Messages {
		if !includeSystem && message.Role == provider.RoleSystem {
			continue
		}

		builder.WriteString(message.Content)
	}

	return builder.String()
}

/*
Reset the context to start fresh
*/
func (ctx *Context) Reset() {
	ctx.iterations = append(ctx.iterations, make([]*provider.Message, 0))
}

/*
AddMessage adds a new message to the context.
*/
func (ctx *Context) AddMessage(message *provider.Message) {
	markdown := markymark.NewDown()

	if message.Role == provider.RoleUser {
		tmpl := viper.GetViper().GetString("prompts.templates.tasks.user")
		message.Content = strings.ReplaceAll(tmpl, "<{user}>", markdown.Quote(message.Content))

		ctx.user = message
		return
	}

	ctx.AddIteration(message.Content)
}

/*
AddIteration adds a new iteration to the context.
*/
func (ctx *Context) AddIteration(response string) {
	if !strings.HasSuffix(response, "\n") {
		response += "\n"
	}

	if ctx.iterations == nil {
		ctx.iterations = make([][]*provider.Message, 0)
	}

	currentIteration := len(ctx.iterations) - 1
	if currentIteration < 0 {
		ctx.iterations = append(ctx.iterations, make([]*provider.Message, 0))
		currentIteration = len(ctx.iterations) - 1
	}

	ctx.iterations[currentIteration] = append(
		ctx.iterations[currentIteration],
		provider.NewMessage(provider.RoleAssistant, response),
	)
}

/*
LastMessage ...
*/
func (ctx *Context) LastMessage() *provider.Message {
	if len(ctx.iterations) == 0 {
		return nil
	}

	return ctx.iterations[len(ctx.iterations)-1][len(ctx.iterations[len(ctx.iterations)-1])-1]
}
