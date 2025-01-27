package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
)

type Ignore struct{}

func NewIgnore() *Ignore {
	return &Ignore{}
}

func (i *Ignore) Execute(
	ctx *drknow.Context,
	args map[string]any,
) Bridge {
	// no-op.
	return nil
}

