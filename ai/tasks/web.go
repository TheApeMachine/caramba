package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
)

type Web struct {
	browser *tools.Browser
}

func NewWeb() *Web {
	return &Web{
		browser: tools.NewBrowser(),
	}
}

func (w *Web) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]any) {
	if err := w.browser.Initialize(); err != nil {
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				err.Error(),
			),
		)
		return
	}

	result := w.browser.Use(ctx.Identity.Ctx, args)

	ctx.AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			result,
		),
	)
}
