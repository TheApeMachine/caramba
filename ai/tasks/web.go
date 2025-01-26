package tasks

import (
	"encoding/json"

	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/tools"
	"github.com/theapemachine/errnie"
)

type Web struct {
}

func NewWeb() *Web {
	return &Web{}
}

func (task *Web) Execute(ctx *drknow.Context, accumulator *stream.Accumulator, args map[string]string) {
	browser := tools.NewBrowser()
	defer browser.Close()

	// Convert string args to map[string]any for the browser
	browserArgs := make(map[string]any)
	for k, v := range args {
		// Handle array values (they come in as JSON strings)
		if v[0] == '[' {
			var arr []string
			if err := json.Unmarshal([]byte(v), &arr); err == nil {
				browserArgs[k] = arr
				continue
			}
		}
		browserArgs[k] = v
	}

	result, err := browser.Run(browserArgs)
	if err != nil {
		errnie.Warn("browser error: %v", err)
		ctx.AddMessage(
			provider.NewMessage(
				provider.RoleAssistant,
				err.Error(),
			),
		)
		return
	}

	ctx.AddMessage(
		provider.NewMessage(
			provider.RoleAssistant,
			result,
		),
	)
}
