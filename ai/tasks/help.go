package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/stream"
)

type Help struct {
}

func NewHelp() *Help {
	return &Help{}
}

func (task *Help) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]string) {

}
