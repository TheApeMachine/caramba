package tasks

import (
	"io"

	"github.com/theapemachine/caramba/ai/drknow"
)

type Task interface {
	Execute(*drknow.Context, map[string]any) Bridge
}

type Bridge interface {
	io.ReadWriteCloser
}
