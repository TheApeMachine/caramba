package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/types"
	"github.com/theapemachine/caramba/utils"
)

type Send struct {
	Tool string `json:"tool" jsonschema:"title=Tool,description=The tool to send messages to agents,enum=send,required"`
	Args struct {
		From    string `json:"from" jsonschema:"title=From,description=The name of the agent sending the message,required"`
		To      string `json:"to" jsonschema:"title=To,description=The name of the agent to send the message to,required"`
		ReplyTo string `json:"reply_to" jsonschema:"title=Reply To,description=The name of the agent to reply to,required"`
		Subject string `json:"subject" jsonschema:"title=Subject,description=The subject of the message,required"`
		Message string `json:"message" jsonschema:"title=Message,description=The message to send to the agent,required"`
	} `json:"args" jsonschema:"title=Arguments,description=The arguments to pass to the tool,required"`
}

func NewSend() *Send {
	return &Send{
		Tool: "send",
	}
}

func (send *Send) Name() string {
	return "send"
}

func (send *Send) Description() string {
	return "Send a message to an existing agent"
}

func (send *Send) GenerateSchema() any {
	return utils.GenerateSchema[Send]()
}

/*
Use sends a message from the first generator (agent) any other
generators passed in after that.
It then takes the responses from the receiving generators, and
appends them to the accumulator, owned by the first generator.
*/
func (send *Send) Use(
	accumulator *stream.Accumulator,
	input map[string]any,
	generators ...types.Generator,
) *stream.Accumulator {
	for _, generator := range generators[1:] {
		accumulator.Generate(generator.Generate(provider.NewMessage(
			provider.RoleUser,
			send.renderMessage(input),
		)))
	}

	return accumulator
}

func (send *Send) Connect(ctx context.Context, rwc io.ReadWriteCloser) error {
	return nil
}

func (send *Send) renderMessage(input map[string]any) string {
	return utils.QuickWrap("MESSAGE", utils.JoinWith("\n",
		"From    : "+input["from"].(string),
		"To      : "+input["to"].(string),
		"Reply-To: "+input["reply_to"].(string),
		"Subject : "+input["subject"].(string),
		"Message : "+input["message"].(string),
	), 2)
}
