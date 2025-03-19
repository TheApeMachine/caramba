package tools

import (
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/github"
)

func init() {
	provider.RegisterTool("azure")
}

type Azure struct {
	buffer *stream.Buffer
	client *github.Client
	Schema *provider.Tool
}

func NewAzure() *Azure {
	client := github.NewClient()

	return &Azure{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("azure.Client.buffer")

			if _, err = io.Copy(client, artifact); err != nil {
				return errnie.Error(err)
			}

			if _, err = io.Copy(artifact, client); err != nil {
				return errnie.Error(err)
			}

			return nil
		}),
		client: client,
		Schema: provider.NewTool(
			provider.WithFunction(
				"github",
				"A tool for interacting with GitHub.",
			),
			provider.WithProperty(
				"operation",
				"string",
				"The operation to perform.",
				[]any{
					"create_work_item",
					"update_work_item",
					"get_work_item",
					"list_work_items",
					"create_wiki_page",
					"update_wiki_page",
					"get_wiki_page",
					"list_wiki_pages",
				},
			),
			provider.WithRequired("operation"),
		),
	}
}

func (azure *Azure) Read(p []byte) (n int, err error) {
	errnie.Debug("azure.Read")
	return azure.buffer.Read(p)
}

func (azure *Azure) Write(p []byte) (n int, err error) {
	errnie.Debug("azure.Write")
	return azure.buffer.Write(p)
}

func (azure *Azure) Close() error {
	errnie.Debug("azure.Close")
	return azure.buffer.Close()
}
