package slack

import (
	"os"

	sdk "github.com/slack-go/slack"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
	"github.com/theapemachine/caramba/pkg/stream"
)

type Client struct {
	buffer *stream.Buffer
	conn   *sdk.Client
}

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

func (client *Client) Read(p []byte) (n int, err error) {
	errnie.Debug("github.Client.Read")
	return client.buffer.Read(p)
}

func (client *Client) Write(p []byte) (n int, err error) {
	errnie.Debug("github.Client.Write")
	return client.buffer.Write(p)
}

func (client *Client) Close() error {
	errnie.Debug("github.Client.Close")
	return client.buffer.Close()
}
