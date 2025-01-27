package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
)

type Break struct{}

func NewBreak() *Break {
	return &Break{}
}

func (b *Break) Execute(
	ctx *drknow.Context,
	args map[string]any,
) Bridge {
	// no-op.
	return nil
}
