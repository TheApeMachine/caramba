package tools

import (
	"encoding/json"
	"io"

	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/workflow"
)

type Caller struct {
	buffer   *stream.Buffer
	toolcall *provider.ToolCall
}

func NewCaller() *Caller {
	errnie.Debug("tools.NewCaller")

	toolcall := &provider.ToolCall{}

	return &Caller{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("tools.Caller.buffer")

			// Convert the payload of the artifact to a ToolCall
			if err = artifact.To(toolcall); err != nil {
				return errnie.Error(err)
			}

			artifact.SetMetaValue("toolcall_id", toolcall.ID)

			var tool io.ReadWriteCloser

			switch toolcall.Function.Name {
			case "agent":
				tool = ai.NewAgent()
			case "memory":
				tool = NewMemoryTool()
			case "browser":
				tool = NewBrowser()
			case "editor":
				tool = NewEditorTool()
			case "environment":
				tool = NewEnvironment()
			case "github":
				tool = NewGithub()
			case "azure":
				tool = NewAzure()
			case "trengo":
				tool = NewTrengo()
			case "system":
				tool = NewSystemTool()
			}

			args := map[string]any{}

			if err = json.Unmarshal(
				[]byte(toolcall.Function.Arguments), &args,
			); err != nil {
				return errnie.Error(err, "toolcall.Function.Arguments", toolcall.Function.Arguments)
			}

			for key, value := range args {
				artifact.SetMetaValue(key, value)
			}

			if err = workflow.NewFlipFlop(artifact, tool); err != nil {
				return errnie.Error(err)
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
