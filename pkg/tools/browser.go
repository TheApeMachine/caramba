package tools

import (
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
)

func init() {
	provider.RegisterTool("browser")
}

type Browser struct {
	buffer *stream.Buffer
	Schema *provider.Tool
}

func NewBrowser() *Browser {
	return &Browser{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) error {
			errnie.Debug("tools.Browser.buffer", artifact)
			return nil
		}),
		Schema: provider.NewTool(
			provider.WithFunction(
				"browser",
				"A fully featured Chrome browser.",
			),
			provider.WithProperty("url", "string", "The URL to navigate to.", []any{}),
			provider.WithProperty(
				"operation",
				"string",
				"The operation to perform.",
				[]any{"run", "close"},
			),
			provider.WithProperty(
				"script",
				"string",
				"A JavaScript function to execute, must be an anonymous function that returns a string, e.g. `() => 'Hello, world!'`.",
				[]any{},
			),
		),
	}
}

func (browser *Browser) Read(p []byte) (n int, err error) {
	errnie.Debug("tools.Browser.Read")
	return browser.buffer.Read(p)
}

func (browser *Browser) Write(p []byte) (n int, err error) {
	errnie.Debug("tools.Browser.Write")
	return browser.buffer.Write(p)
}

func (browser *Browser) Close() error {
	errnie.Debug("tools.Browser.Close")
	return browser.buffer.Close()
}
