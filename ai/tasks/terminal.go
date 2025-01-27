package tasks

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/errnie"
)

// Terminal implements a Task that connects to an isolated Debian-based
// environment and provides an interactive shell via the container tool.
type Terminal struct{}

// NewTerminal creates a new Terminal task.
func NewTerminal() Task {
	return &Terminal{}
}

// Execute locates the Container tool, ensures it is initialized, and
// returns a Bridge that streams user input/output to the container shell.
func (t *Terminal) Execute(ctx *drknow.Context, args map[string]any) Bridge {
	// Find the first Container tool in the agent's tool list
	containerTool := tools.NewContainer()

	// Initialize the container if needed
	if err := containerTool.Initialize(); err != nil {
		errnie.Error(err)
		return nil
	}

	// If not already connected, connect now
	if containerTool.Conn == nil {
		if err := containerTool.Connect(context.Background(), nil); err != nil {
			errnie.Error(err)
			return nil
		}
	}

	// Return a simple I/O bridge to pass data between user input and the container
	return &ioBridge{Conn: containerTool.Conn}
}

// ioBridge is a minimal tasks.Bridge implementation around an io.ReadWriteCloser.
type ioBridge struct {
	Conn io.ReadWriteCloser
}

func (b *ioBridge) Read(p []byte) (n int, err error) {
	return b.Conn.Read(p)
}

func (b *ioBridge) Write(p []byte) (n int, err error) {
	return b.Conn.Write(p)
}

func (b *ioBridge) Close() error {
	return b.Conn.Close()
}
