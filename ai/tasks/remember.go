package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/stream"
)

type Remember struct {
}

func NewRemember() *Remember {
	return &Remember{}
}

func (task *Remember) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]string) {

}
