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

/*
init registers the GitHub tool with the provider system.
*/
func init() {
	fmt.Println("tools.github.init")
	provider.RegisterTool("github")
}

/*
Github provides a streaming interface to GitHub operations.
It manages GitHub API interactions through a buffered client connection
and implements io.ReadWriteCloser for streaming data processing.
*/
type Github struct {
	buffer *stream.Buffer
	client *github.Client
	Schema *provider.Tool
}

/*
NewGithub creates a new GitHub tool instance.

It initializes a GitHub client and sets up a buffered stream for
processing GitHub operations. The buffer copies data bidirectionally
between the artifact and the GitHub client.
*/
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

/*
Read implements the io.Reader interface.

It reads processed data from the internal buffer after GitHub operations
have been completed.
*/
func (github *Github) Read(p []byte) (n int, err error) {
	errnie.Debug("github.Read")
	return github.buffer.Read(p)
}

/*
Write implements the io.Writer interface.

It writes operation requests to the internal buffer for processing
by the GitHub client.
*/
func (github *Github) Write(p []byte) (n int, err error) {
	errnie.Debug("github.Write")
	return github.buffer.Write(p)
}

/*
Close implements the io.Closer interface.

It cleans up resources by closing the internal buffer.
*/
func (github *Github) Close() error {
	errnie.Debug("github.Close")
	return github.buffer.Close()
}
