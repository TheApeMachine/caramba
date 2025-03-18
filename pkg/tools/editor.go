package tools

import (
	"os"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
)

func init() {
	provider.RegisterTool("editor")
}

type EditorTool struct {
	buffer *stream.Buffer
	fh     *os.File
	Schema *provider.Tool
}

func NewEditorTool() *EditorTool {
	return &EditorTool{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("EditorTool.buffer")
			return nil
		}),
		Schema: provider.NewTool(
			provider.WithFunction(
				"editor",
				"A tool which can edit a file.",
			),
			provider.WithProperty(
				"file",
				"string",
				"The file to edit.",
				[]any{},
			),
			provider.WithProperty(
				"operation",
				"string",
				"The operation to perform.",
				[]any{"read", "write", "delete"},
			),
			provider.WithProperty(
				"content",
				"string",
				"The content to write to the file.",
				[]any{},
			),
			provider.WithProperty(
				"start",
				"number",
				"Scope the operation to be applied from this line number.",
				[]any{},
			),
			provider.WithProperty(
				"end",
				"number",
				"Scope the operation to be applied up to this line number.",
				[]any{},
			),
		),
	}
}

func (et *EditorTool) Read(p []byte) (n int, err error) {
	errnie.Debug("EditorTool.Read")
	return et.buffer.Read(p)
}

func (et *EditorTool) Write(p []byte) (n int, err error) {
	errnie.Debug("EditorTool.Write")
	return et.buffer.Write(p)
}

func (et *EditorTool) Close() error {
	errnie.Debug("EditorTool.Close")
	return et.buffer.Close()
}
