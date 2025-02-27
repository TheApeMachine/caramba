package tools

import (
	"context"

	"github.com/theapemachine/caramba/pkg/agent/tools/browser"
)

// BrowserToolWrapper is a wrapper for the browser tool that implements the Tool interface
type BrowserTool struct {
	browserTool *browser.Tool
}

// NewBrowserTool creates a new BrowserTool instance
func NewBrowserTool(browserlessURL string, browserlessAPIKey string) *BrowserTool {
	return &BrowserTool{
		browserTool: browser.New(browserlessURL, browserlessAPIKey),
	}
}

// Name returns the name of the tool
func (b *BrowserTool) Name() string {
	return b.browserTool.Name()
}

// Description returns the description of the tool
func (b *BrowserTool) Description() string {
	return b.browserTool.Description()
}

// Schema returns the JSON schema for the tool's arguments
func (b *BrowserTool) Schema() map[string]interface{} {
	return b.browserTool.Schema()
}

// Execute executes the tool with the given arguments
func (b *BrowserTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	return b.browserTool.Execute(ctx, args)
}
