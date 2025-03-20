package tools

import (
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/browser"
)

func init() {
	provider.RegisterTool("browser")
}

type Browser struct {
	buffer   *stream.Buffer
	Schema   *provider.Tool
	instance *browser.Instance
}

func NewBrowser() *Browser {
	instance := browser.NewInstance()

	return &Browser{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("tools.Browser.buffer")
			url := datura.GetMetaValue[string](artifact, "url")
			script := datura.GetMetaValue[string](artifact, "script")

			if url == "" {
				datura.WithPayload([]byte("url is required"))(artifact)
			}

			if script == "" {
				datura.WithPayload([]byte("script is required"))(artifact)
			}

			if _, err = io.Copy(instance, artifact); err != nil {
				return errnie.Error(err)
			}

			if _, err = io.Copy(artifact, instance); err != nil {
				return errnie.Error(err)
			}

			return nil
		}),
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
