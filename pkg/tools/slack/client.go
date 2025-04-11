package slack

import (
	"os"

	sdk "github.com/slack-go/slack"
)

/*
Client provides a high-level interface to Slack services.
It manages connections and operations for channels and messages
through a unified streaming interface.
*/
type Client struct {
	conn *sdk.Client
}

/*
NewClient creates a new Slack client using environment variables for authentication.

It initializes the connection using a bot token from the MARVIN_BOT_TOKEN
environment variable and sets up a buffered stream for processing operations.
*/
func NewClient() *Client {
	client := sdk.New(os.Getenv("MARVIN_BOT_TOKEN"))

	return &Client{
		conn: client,
	}
}

func (client *Client) Generate(
	buffer chan map[string]any,
	fn ...func(artifact map[string]any) map[string]any,
) chan map[string]any {
	out := make(chan map[string]any)

	go func() {
		defer close(out)
	}()

	return out
}
