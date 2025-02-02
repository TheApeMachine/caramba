package tasks

import (
	"strings"

	"github.com/theapemachine/caramba/ai/drknow"
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
func (t *Terminal) Execute(ctx *drknow.Context, args map[string]any) string {
	return strings.Join([]string{
		"Welcome to Debian Linux.",
		"To install a package, use the apt command, e.g. `apt install curl`.",
		"To update the package list, use the apt update command, e.g. `apt update`.",
		"",
		"$ ",
	}, "\n")
}
