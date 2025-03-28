package tools

import (
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/github"
)

func init() {
	fmt.Println("tools.azure.init")
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
		Schema: GetToolSchema("azure"),
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
