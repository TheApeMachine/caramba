package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/errnie"
)

type Web struct {
}

func NewWeb() *Web {
	return &Web{}
}

func (task *Web) Execute(ctx *drknow.Context, args map[string]any) Bridge {
	browser := tools.NewBrowser()
	defer browser.Close()

	result, err := browser.Run(args)
	if err != nil {
		errnie.Warn("browser error: %v", err)
	}

	ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, result))

	return nil
}
