package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/stream"
)

type Break struct{}

func NewBreak() *Break {
	return &Break{}
}

func (b *Break) Execute(
	ctx *drknow.Context,
	accumulator *stream.Accumulator,
	args map[string]any,
) {
	// no-op.
}
