package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/stream"
)

type Ignore struct{}

func NewIgnore() *Ignore {
	return &Ignore{}
}

func (i *Ignore) Execute(
	ctx *drknow.Context,
	accumulator *stream.Accumulator,
	args map[string]any,
) {
	// no-op.
}
