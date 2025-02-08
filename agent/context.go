package agent

import (
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tweaker"
)

type Context struct {
	config    *Config
	params    *provider.LLMGenerationParams
	iteration int
}

func NewContext(config *Config) *Context {
	return &Context{
		config:    config,
		params:    provider.NewGenerationParams(config.Thread),
		iteration: 0,
	}
}

func (ctx *Context) AddUserPrompt(
	message *provider.Message,
) {
	ctx.config.Thread.AddMessage(
		provider.NewMessage(
			provider.RoleUser,
			tweaker.GetUserPrompt(message.Content),
		),
	)

	ctx.config.Thread.AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			tweaker.GetContext(),
		),
	)
}
