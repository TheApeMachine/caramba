package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

type HumanMode uint

const (
	HumanUser HumanMode = iota
	HumanProxy
)

/*
Human is a tool wrapper that has two distinct modes:

 1. It can be configured to call back to the human user to ask for intermediate input.
 2. It can be configured to act as a human user proxy, where an agent will perform
    the role of the human user.
*/
type Human struct {
	mode    HumanMode
	Message string `json:"message" jsonschema:"title=Message,description=The message to send to the human user,required"`
}

func NewHuman(mode HumanMode) *Human {
	return &Human{
		mode: mode,
	}
}

func (human *Human) Name() string {
	return "human"
}

func (human *Human) Description() string {
	return "Interact with a human user"
}

func (human *Human) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Human]()
}

func (human *Human) Use(ctx context.Context, params map[string]any) string {
	return ""
}

func (human *Human) Connect(ctx context.Context, conn io.ReadWriteCloser) error {
	return nil
}
