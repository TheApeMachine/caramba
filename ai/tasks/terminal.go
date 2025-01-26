package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/stream"
)

type Terminal struct {
}

func NewTerminal() *Terminal {
	return &Terminal{}
}

func (task *Terminal) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]string) {

}
