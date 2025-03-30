package slack

import (
	"os"

	sdk "github.com/slack-go/slack"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

/*
Client provides a high-level interface to Slack services.
It manages connections and operations for channels and messages
through a unified streaming interface.
*/
type Client struct {
	buffer *stream.Buffer
	conn   *sdk.Client
}

/*
NewClient creates a new Slack client using environment variables for authentication.

It initializes the connection using a bot token from the MARVIN_BOT_TOKEN
environment variable and sets up a buffered stream for processing operations.
*/
func NewClient() *Client {
	client := sdk.New(os.Getenv("MARVIN_BOT_TOKEN"))

	return &Client{
		buffer: stream.NewBuffer(func(artifact *datura.Artifact) (err error) {
			errnie.Debug("github.Client.buffer")

			operation := datura.GetMetaValue[string](artifact, "operation")

			switch operation {
			case "send_message":

			}

			return nil
		}),
		conn: client,
	}
}

/*
Read implements the io.Reader interface.

It reads processed data from the internal buffer after Slack operations
have been completed.
*/
func (client *Client) Read(p []byte) (n int, err error) {
	errnie.Debug("github.Client.Read")
	return client.buffer.Read(p)
}

/*
Write implements the io.Writer interface.

It writes operation requests to the internal buffer for processing by
the appropriate Slack service.
*/
func (client *Client) Write(p []byte) (n int, err error) {
	errnie.Debug("github.Client.Write")
	return client.buffer.Write(p)
}

/*
Close implements the io.Closer interface.

It cleans up resources by closing the internal buffer.
*/
func (client *Client) Close() error {
	errnie.Debug("github.Client.Close")
	return client.buffer.Close()
}
