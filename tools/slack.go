package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

/*
Slack wraps the Slack API and turns it into a tool for agents to use.
As a special case, this tool needs to also bring up a server to listen
for incoming events.
*/
type Slack struct {
	Operation string `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=send_message,enum=search,required"`
	Channel   string `json:"channel" jsonschema:"title=Channel,description=The channel to send the message to"`
	Message   string `json:"message" jsonschema:"title=Message,description=The message to send"`
}

func NewSlack() *Slack {
	return &Slack{}
}

func (slack *Slack) Name() string {
	return "slack"
}

func (slack *Slack) Description() string {
	return "Interact with Slack"
}

func (slack *Slack) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Slack]()
}

func (slack *Slack) Use(ctx context.Context, params map[string]any) string {
	return ""
}

func (slack *Slack) Connect(ctx context.Context, rw io.ReadWriteCloser) error {
	return nil
}
