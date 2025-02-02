package tasks

import (
	"github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/tools"
)

type Web struct {
	browser *tools.Browser
}

func NewWeb() *Web {
	return &Web{
		browser: tools.NewBrowser(),
	}
}

func (task *Web) Execute(ctx *drknow.Context, args map[string]any) string {
	log.Info("Starting Web task", "args", args)

	result, err := task.browser.Run(args)
	if err != nil {
		log.Error(err)
		return err.Error()
	}

	if result != "" {
		return result
	}

	return "No content could be extracted from the page"
}
