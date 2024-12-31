package ai

import (
	"strconv"
	"strings"

	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/utils"
)

/*
Context is a wrapper that manages the current Thread, dealing with structural
formatting, and making sure that everything is aligned correctly, even in
nested tags.
*/
type Context struct {
	params     *provider.GenerationParams
	Thread     *provider.Thread
	Scratchpad *provider.Thread
	System     *System
	indent     int
}

/*
NewContext takes a Thread and wraps it in a Context, so that it can be used
to manage the various messages with the appropriate formatting.
*/
func NewContext(system *System, params *provider.GenerationParams) *Context {
	return &Context{
		params:     params,
		Thread:     params.Thread,
		Scratchpad: provider.NewThread(),
		System:     system,
		indent:     0,
	}
}

/*
Compile the current context to curate the messages in the thread, so they are
all formatted correctly, in the right order, and with the correct roles.
*/
func (ctx *Context) Compile() *provider.GenerationParams {
	ctx.Thread.Reset()

	ctx.Thread.AddMessage(provider.NewMessage(
		provider.RoleSystem,
		utils.QuickWrap(
			"system",
			utils.Substitute(ctx.System.fragments["prompt"], ctx.System.fragments, ctx.indent),
			ctx.indent,
		),
	))

	userCount := 0

	for _, message := range ctx.Scratchpad.Messages {
		switch message.Role {
		case provider.RoleUser:
			tag := "goal"

			if userCount > 0 {
				tag = "subtask"
			}

			ctx.Thread.AddMessage(provider.NewMessage(
				provider.RoleUser,
				utils.QuickWrap(tag, utils.Reflow(utils.StripXML(message.Content)), ctx.indent),
			))

			userCount++
		case provider.RoleAssistant:
			iteration := 0

			ctx.Thread.AddMessage(provider.NewMessage(
				provider.RoleAssistant,
				utils.QuickWrapWithAttributes(
					"response",
					utils.Reflow(utils.StripXML(message.Content)),
					ctx.indent,
					map[string]string{
						"agent":     ctx.System.fragments["name"],
						"role":      ctx.System.fragments["role"],
						"iteration": strconv.Itoa(iteration),
					},
				),
			))

			iteration++
		}
	}

	ctx.params.Thread = ctx.Thread
	return ctx.params
}

/*
Scratchpad returns a fresh scratchpad, so we can accumulate messages before
compiling.
*/
func (ctx *Context) GetScratchpad() *Context {
	ctx.Scratchpad = provider.NewThread()
	return ctx
}

/*
Append to the context, so we can accumulate messages before compiling.
*/
func (ctx *Context) Append(event provider.Event) {
	ctx.Scratchpad.Messages[len(ctx.Scratchpad.Messages)-1].Content += event.Text
}

/*
ToolCall deals with the tool calls, so we can accumulate them in the scratchpad
before compiling.
*/
func (ctx *Context) ToolCall(event provider.Event) {
	ctx.Scratchpad.AddMessage(provider.NewMessage(
		provider.RoleTool,
		utils.QuickWrap("tool", event.Text, ctx.indent),
	))
}

/*
Done checks the current scratchpad to see if the agent has indicated that it
wants to stop iteration.
*/
func (ctx *Context) Done() bool {
	for _, message := range ctx.Scratchpad.Messages {
		if strings.Contains(message.Content, "[STOP]") {
			return true
		}
	}

	return false
}
