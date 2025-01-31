package tasks

import (
	"fmt"

	"github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
)

type Web struct {
}

func NewWeb() *Web {
	return &Web{}
}

func (task *Web) Execute(ctx *drknow.Context, args map[string]any) Bridge {
	log.Info("Starting Web task", "args", args)

	browser := tools.NewBrowser()
	defer browser.Close()

	// Validate required URL parameter
	url, ok := args["url"].(string)
	if !ok || url == "" {
		errMsg := "URL is required"
		log.Error(errMsg)
		ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, errMsg))
		return nil
	}

	// First handle scroll if requested
	if action, ok := args["action"].(string); ok && action == "scroll" {
		scrollArgs := map[string]any{
			"url":    url,
			"action": "scroll",
		}
		if _, err := browser.Run(scrollArgs); err != nil {
			errMsg := fmt.Sprintf("Scroll error: %v", err)
			log.Error(errMsg)
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, errMsg))
			return nil
		}
	}

	// Extract content using provided JavaScript
	if js, ok := args["javascript"].(string); ok && js != "" {
		contentArgs := map[string]any{
			"url":        url,
			"javascript": js,
		}
		result, err := browser.Run(contentArgs)
		if err != nil {
			errMsg := fmt.Sprintf("Browser error: %v", err)
			log.Error(errMsg)
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, errMsg))
			return nil
		}

		if result == "" {
			log.Warn("No content extracted from page")
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, "No content could be extracted from the page"))
			return nil
		}

		// Format the result nicely
		formattedResult := fmt.Sprintf("Content from %s:\n%s", url, result)
		ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, formattedResult))
	} else {
		errMsg := "JavaScript selector is required for content extraction"
		log.Error(errMsg)
		ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, errMsg))
	}

	return nil
}
