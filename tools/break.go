package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

type Break struct {
	Tool        string `json:"tool" jsonschema:"title=Tool,description=The tool to break out of your iteration loop,enum=break,required"`
	FinalAnswer string `json:"final_answer" jsonschema:"title=Final Answer,description=Optionally add a finaly answer, synthesized from your iterations"`
}

func NewBreak() *Break {
	return &Break{
		Tool: "break",
	}
}

func (brk *Break) Name() string {
	return "break"
}

func (brk *Break) Description() string {
	return "Show information about the system state"
}

func (brk *Break) GenerateSchema() interface{} {
	return utils.GenerateSchema[Show]()
}

func (brk *Break) Use(input map[string]any) string {
	return "break"
}

func (brk *Break) Connect(ctx context.Context, rwc io.ReadWriteCloser) error {
	return nil
}
