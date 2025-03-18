package tools

import (
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/tools/browser"
)

func init() {
	provider.RegisterTool("browser")
}

type Browser struct {
	Schema   *provider.Tool
	instance *browser.Instance
}

func NewBrowser() *Browser {
	instance := browser.NewInstance()

	return &Browser{
		Schema: provider.NewTool(
			provider.WithFunction(
				"browser",
				"A fully featured Chrome browser.",
			),
			provider.WithProperty("url", "string", "The URL to navigate to.", []any{}),
			provider.WithProperty(
				"script",
				"string",
				"A JavaScript function to execute, must be an anonymous function that returns a string, e.g. `() => 'Hello, world!'`.",
				[]any{},
			),
			provider.WithRequired("url", "script"),
		),
		instance: instance,
	}
}

func (browser *Browser) Read(p []byte) (n int, err error) {
	errnie.Debug("tools.Browser.Read")
	return browser.instance.Read(p)
}

func (browser *Browser) Write(p []byte) (n int, err error) {
	errnie.Debug("tools.Browser.Write")
	return browser.instance.Write(p)
}

func (browser *Browser) Close() error {
	errnie.Debug("tools.Browser.Close")
	return browser.instance.Close()
}
