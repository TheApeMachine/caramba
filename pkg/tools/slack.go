package tools

import (
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/slack"
)

func init() {
	provider.RegisterTool("slack")
}

type Slack struct {
	buffer *stream.Buffer
	Schema *provider.Tool
	client *slack.Client
}

func NewSlack() *Slack {
	client := slack.NewClient()

	return &Slack{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("slack.Client.buffer")

			if _, err = io.Copy(client, artifact); err != nil {
				return errnie.Error(err)
			}

			if _, err = io.Copy(artifact, client); err != nil {
				return errnie.Error(err)
			}

			return nil
		}),
		client: client,
		Schema: GetToolSchema("slack"),
	}
}

func (s *Slack) Read(p []byte) (n int, err error) {
	errnie.Debug("slack.Read")
	return s.buffer.Read(p)
}

func (s *Slack) Write(p []byte) (n int, err error) {
	errnie.Debug("slack.Write")
	return s.buffer.Write(p)
}

func (s *Slack) Close() error {
	errnie.Debug("slack.Close")
	return s.buffer.Close()
}
