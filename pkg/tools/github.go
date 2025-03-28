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
	fmt.Println("tools.github.init")
	provider.RegisterTool("github")
}

type Github struct {
	buffer *stream.Buffer
	client *github.Client
	Schema *provider.Tool
}

func NewGithub() *Github {
	client := github.NewClient()

	return &Github{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("github.Client.buffer")

			if _, err = io.Copy(client, artifact); err != nil {
				return errnie.Error(err)
			}

			if _, err = io.Copy(artifact, client); err != nil {
				return errnie.Error(err)
			}

			return nil
		}),
		client: client,
		Schema: GetToolSchema("github"),
	}
}

func (github *Github) Read(p []byte) (n int, err error) {
	errnie.Debug("github.Read")
	return github.buffer.Read(p)
}

func (github *Github) Write(p []byte) (n int, err error) {
	errnie.Debug("github.Write")
	return github.buffer.Write(p)
}

func (github *Github) Close() error {
	errnie.Debug("github.Close")
	return github.buffer.Close()
}
