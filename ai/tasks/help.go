package tasks

import (
	"github.com/theapemachine/caramba/ai/drknow"
)

type Help struct {
}

func NewHelp() *Help {
	return &Help{}
}

func (task *Help) Execute(ctx *drknow.Context, args map[string]any) Bridge {
	return nil
}
