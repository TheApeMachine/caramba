package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/errnie"
)

type Web struct {
}

func NewWeb() *Web {
	return &Web{}
}

func (task *Web) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]any) {
	browser := tools.NewBrowser()
	defer browser.Close()

	result, err := browser.Run(args)
	if err != nil {
		errnie.Warn("browser error: %v", err)
	}

	accumulator.Write([]byte(result))
}
