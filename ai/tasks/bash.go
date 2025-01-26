package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/stream"
)

type Bash struct {
}

func NewBash() *Bash {
	return &Bash{}
}

func (task *Bash) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]string) {

}
