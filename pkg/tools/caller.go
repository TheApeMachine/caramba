package tools

import (
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Function struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ToolCall struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Caller struct {
	buffer   *stream.Buffer
	toolcall *ToolCall
}

func NewCaller() *Caller {
	errnie.Debug("tools.NewCaller")
	toolcall := &ToolCall{}

	return &Caller{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("tools.Caller.buffer")

			var payload []byte

			if payload, err = artifact.DecryptPayload(); err != nil {
				return errnie.Error(err)
			}

			if err = json.Unmarshal(payload, toolcall); err != nil {
				return errnie.Error(err)
			}

			args := map[string]any{}

			if err = json.Unmarshal([]byte(toolcall.Function.Arguments), &args); err != nil {
				return errnie.Error(err)
			}

			switch toolcall.Function.Name {
			case "browser":
				browser := NewBrowser()

				var (
					ok     bool
					url    string
					script string
				)

				if url, ok = args["url"].(string); ok {
					artifact.SetMetaValue("url", url)
				}

				if script, ok = args["script"].(string); ok {
					artifact.SetMetaValue("script", script)
				}

				if _, err = io.Copy(browser, artifact); err != nil {
					return errnie.Error(err)
				}

				if _, err = io.Copy(artifact, browser); err != nil {
					return errnie.Error(err)
				}
			}

			return nil
		}),
		toolcall: toolcall,
	}
}

func (caller *Caller) Read(p []byte) (n int, err error) {
	errnie.Debug("tools.Caller.Read")
	return caller.buffer.Read(p)
}

func (caller *Caller) Write(p []byte) (n int, err error) {
	errnie.Debug("tools.Caller.Write")
	return caller.buffer.Write(p)
}

func (caller *Caller) Close() error {
	errnie.Debug("tools.Caller.Close")
	return caller.buffer.Close()
}
