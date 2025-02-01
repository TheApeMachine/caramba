package tasks

import (
	"context"

	"github.com/theapemachine/caramba/ai/drknow"
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

func (task *Bash) Execute(ctx *drknow.Context, args map[string]any) string {
	task.container.Initialize(context.Background())
	response, err := task.container.RunCommandInteractive(context.Background(), args["command"].(string))
	if err != nil {
		return err.Error()
	}
	ctx.AddIteration(response)

	return response
}
