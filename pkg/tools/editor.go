package tools

import (
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/editor"
)

func init() {
	provider.RegisterTool("editor")
}

type EditorTool struct {
	buffer *stream.Buffer
	client *editor.Client
	Schema *provider.Tool
}

func NewEditorTool() *EditorTool {
	client := editor.NewClient()

	return &EditorTool{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("editor.Client.buffer")

			operation := datura.GetMetaValue[string](artifact, "operation")

			switch operation {
			case "read":
				return client.ReadFile(artifact)
			case "write":
				return client.WriteFile(artifact)
			case "delete":
				return client.DeleteFile(artifact)
			case "replace_lines":
				return client.ReplaceLines(artifact)
			case "insert_lines":
				return client.InsertLines(artifact)
			case "delete_lines":
				return client.DeleteLines(artifact)
			case "read_lines":
				return client.ReadLines(artifact)
			}

			return nil
		}),
		client: client,
		Schema: provider.NewTool(
			provider.WithFunction(
				"editor",
				"A tool which can edit files in the workspace with special support for code editing.",
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
				[]any{
					"read",
					"write",
					"delete",
					"replace_lines",
					"insert_lines",
					"delete_lines",
					"read_lines",
				},
			),
			provider.WithProperty(
				"content",
				"string",
				"The content to write to the file. For line operations, this can be multiple lines separated by newlines.",
				[]any{},
			),
			provider.WithProperty(
				"start_line",
				"number",
				"The line number to start the operation from (1-based indexing).",
				[]any{},
			),
			provider.WithProperty(
				"end_line",
				"number",
				"The line number to end the operation at (1-based indexing, inclusive).",
				[]any{},
			),
			provider.WithProperty(
				"line_number",
				"number",
				"The line number for single-line operations like insertion (1-based indexing).",
				[]any{},
			),
			provider.WithRequired("file", "operation"),
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
