package tools

import (
	"context"

	"github.com/theapemachine/caramba/pkg/agent/tools/browser"
)

type BrowserTool struct {
	browserTool *browser.Tool
}

func NewBrowserTool(browserlessURL string, browserlessAPIKey string) *BrowserTool {
	return &BrowserTool{
		browserTool: browser.New(browserlessURL, browserlessAPIKey),
	}
}

func (b *BrowserTool) Name() string {
	return b.browserTool.Name()
}

func (b *BrowserTool) Description() string {
	return b.browserTool.Description()
}

func (b *BrowserTool) Schema() map[string]interface{} {
	return b.browserTool.Schema()
}

func (b *BrowserTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	return b.browserTool.Execute(ctx, args)
}
