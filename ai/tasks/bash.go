package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/stream"
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

func (task *Bash) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]any) {
	if err := task.container.Initialize(); err != nil {
		accumulator.Write([]byte(err.Error()))
		return
	}

	result := task.container.Use(ctx.Identity.Ctx, args)
	accumulator.Write([]byte(result))
}
