package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/stream"
)

type Task interface {
	Execute(*drknow.Context, *stream.Accumulator, map[string]any)
}
