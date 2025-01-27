package tasks

import (
	"io"

	"github.com/theapemachine/caramba/ai/drknow"
)

/*
Task represents an interface for executing a specific task.
It provides a way to start and execute a task, and it returns a
Bridge for interacting with the task.
*/
type Task interface {
	Execute(*drknow.Context, map[string]any) Bridge
}

type Bridge interface {
	io.ReadWriteCloser
	Start()
}
