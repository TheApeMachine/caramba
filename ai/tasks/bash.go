package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
)

type Bash struct {
	container *tools.Container
}

func NewBash() *Bash {
	return &Bash{
		container: tools.NewContainer(),
	}
}

func (task *Bash) Execute(ctx *drknow.Context, args map[string]any) Bridge {
	task.container.Initialize()
	response := task.container.ExecuteCommand(args["command"].(string))
	ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, response))
	return nil
}
