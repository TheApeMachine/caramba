package tasks

import (
	"strings"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
)

type Bash struct {
	container *tools.Container
}

func NewBash() *Bash {
	return &Bash{
		container: tools.NewContainer(),
	}
}

func (task *Bash) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]any) {
	if err := task.container.Initialize(); err != nil {
		accumulator.Write([]byte(err.Error()))
		return
	}

	// Extract commands array and join into single command
	cmds, ok := args["commands"].([]any)
	if !ok {
		accumulator.Write([]byte("error: invalid commands parameter"))
		return
	}

	// Convert commands to strings and join with &&
	cmdStrings := make([]string, len(cmds))
	for i, cmd := range cmds {
		if cmdStr, ok := cmd.(string); ok {
			cmdStrings[i] = cmdStr
		}
	}
	command := strings.Join(cmdStrings, " && ")

	// Pass single command to container
	result := task.container.Use(ctx.Identity.Ctx, map[string]any{
		"command": command,
	})

	ctx.AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			result,
		),
	)
}
