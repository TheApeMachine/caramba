package tools

import (
	"fmt"
	"io"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/provider"
	"github.com/theapemachine/caramba/pkg/stream"
	"github.com/theapemachine/caramba/pkg/tools/slack"
)

/*
init registers the Slack tool with the provider system.
*/
func init() {
	fmt.Println("tools.slack.init")
	provider.RegisterTool("slack")
}

/*
Slack provides a streaming interface to Slack operations.
It manages Slack API interactions through a buffered client connection
and implements io.ReadWriteCloser for streaming data processing.
*/
type Slack struct {
	buffer *stream.Buffer
	Schema *provider.Tool
	client *slack.Client
}

/*
NewSlack creates a new Slack tool instance.

It initializes a Slack client and sets up a buffered stream for
processing Slack operations. The buffer copies data bidirectionally
between the artifact and the Slack client.
*/
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

/*
Read implements the io.Reader interface.

It reads processed data from the internal buffer after Slack operations
have been completed.
*/
func (s *Slack) Read(p []byte) (n int, err error) {
	errnie.Debug("slack.Read")
	return s.buffer.Read(p)
}

/*
Write implements the io.Writer interface.

It writes operation requests to the internal buffer for processing
by the Slack client.
*/
func (s *Slack) Write(p []byte) (n int, err error) {
	errnie.Debug("slack.Write")
	return s.buffer.Write(p)
}

/*
Close implements the io.Closer interface.

It cleans up resources by closing the internal buffer.
*/
func (s *Slack) Close() error {
	errnie.Debug("slack.Close")
	return s.buffer.Close()
}
