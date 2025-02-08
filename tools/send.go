package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

type Send struct {
	Tool      string `json:"tool" jsonschema:"title=Tool,description=The tool to send messages to agents,enum=send,required"`
	AgentName string `json:"name" jsonschema:"title=Name,description=The name of the agent to send the message to,required"`
	Message   string `json:"message" jsonschema:"title=Message,description=The message to send to the agent,required"`
}

func NewSend() *Send {
	return &Send{
		Tool: "send",
	}
}

func (s *Send) Name() string {
	return "send"
}

func (s *Send) Description() string {
	return "Send a message to an existing agent"
}

func (s *Send) GenerateSchema() interface{} {
	return utils.GenerateSchema[Send]()
}

func (s *Send) Use(input map[string]any) string {
	return "send"
}

func (s *Send) Connect(ctx context.Context, rwc io.ReadWriteCloser) error {
	return nil
}
