package rod

import (
	"context"
	"fmt"
	"time"

	"github.com/go-rod/rod"
	"github.com/theapemachine/caramba/pkg/hub"
	"github.com/theapemachine/caramba/pkg/output"
)

// Tool represents the Browser Tool for web interactions using Go Rod
type Tool struct {
	logger  *output.Logger
	hub     *hub.Queue
	browser *rod.Browser
	timeout time.Duration
	page    *rod.Page
}

// New creates a new Rod Browser Tool instance
func New() *Tool {
	timeout := 30 * time.Second

	return &Tool{
		logger:  output.NewLogger(),
		hub:     hub.NewQueue(),
		timeout: timeout,
	}
}

// Name returns the name of the tool
func (t *Tool) Name() string {
	return "rod-browser"
}

// Description returns the description of the tool
func (t *Tool) Description() string {
	return "A tool for interacting with web pages directly using Go Rod, including navigation, taking screenshots, and extracting content."
}

// Schema returns the schema for the tool in JSON format
func (t *Tool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"action": map[string]any{
				"type":        "string",
				"enum":        []string{"navigate", "screenshot", "pdf", "extract", "execute", "search"},
				"description": "Action to perform with the browser tool",
			},
			"url": map[string]any{
				"type":        "string",
				"description": "URL to navigate to (required for navigate, screenshot, pdf, extract, optional for execute)",
			},
			"query": map[string]any{
				"type":        "string",
				"description": "Search query (required for search action)",
			},
			"selector": map[string]any{
				"type":        "string",
				"description": "CSS selector for content extraction (required for extract)",
			},
			"script": map[string]any{
				"type":        "string",
				"description": "JavaScript code to execute (required for execute)",
			},
			"wait_for": map[string]any{
				"type":        "string",
				"description": "CSS selector to wait for before performing the action",
			},
			"full_page": map[string]any{
				"type":        "boolean",
				"description": "Capture full page for screenshot/PDF (default: false)",
			},
			"timeout": map[string]any{
				"type":        "integer",
				"description": "Operation timeout in seconds (default: 30)",
			},
		},
		"required": []string{"action"},
	}
}

// Execute executes the browser tool with the given action and arguments
func (t *Tool) Execute(ctx context.Context, args map[string]any) (any, error) {
	action, ok := args["action"].(string)
	if !ok {
		return nil, t.logger.Error(t.Name(), fmt.Errorf("action parameter is required"))
	}

	var result any
	var err error

	switch action {
	case "navigate":
		result, err = t.navigate(ctx, args)
	case "screenshot":
		result, err = t.screenshot(ctx, args)
	case "extract":
		result, err = t.extractContent(ctx, args)
	case "search":
		result, err = t.search(ctx, args)
	case "pdf":
		result, err = t.generatePDF(ctx, args)
	case "execute":
		result, err = t.executeScript(ctx, args)
	default:
		return nil, t.logger.Error(t.Name(), fmt.Errorf("unknown action: %s", action))
	}

	if err != nil {
		return nil, t.logger.Error(t.Name(), err)
	}

	return result, nil
}

// Close cleans up resources
func (t *Tool) Close() error {
	if t.browser != nil {
		return t.browser.Close()
	}

	return nil
}
