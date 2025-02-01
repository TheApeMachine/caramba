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

	// Execute browser operation
	result, err := task.browser.Run(args)
	if err != nil {
		log.Error(err)
		ctx.AddIteration(err.Error())
		return err.Error()
	}

	// Add result to context as assistant message
	if result != "" {
		ctx.AddIteration(result)
		return result
	}

	log.Warn("No content extracted from page")
	ctx.AddIteration("No content could be extracted from the page")

	return "No content could be extracted from the page"
}
