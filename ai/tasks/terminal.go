package tasks

import (
	"context"

	"github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/tools"
)

/*
Terminal implements a Task that connects to an isolated Debian-based
environment and provides an interactive shell via the container tool.
*/
type Terminal struct{}

/*
NewTerminal creates a new Terminal task.
*/
func NewTerminal() Task {
	return &Terminal{}
}

/*
Execute locates the Container tool, ensures it is initialized, and
returns a Bridge that streams user input/output to the container shell.
*/
func (t *Terminal) Execute(ctx *drknow.Context, args map[string]any) Bridge {
	// Find or create our Container tool
	containerTool := tools.NewContainer()

	// Connect if needed
	if containerTool.Conn == nil {
		if err := containerTool.Connect(context.Background(), nil); err != nil {
			log.Error("Error connecting to container", "error", err)
			return nil
		}
	}

	if err := containerTool.Start(); err != nil {
		log.Error("Error starting container", "error", err)
		return nil
	}

	// Return a Bridge wrapping the container I/O
	return &IOBridge{
		container: containerTool,
		Conn:      containerTool.Conn,
	}
}
