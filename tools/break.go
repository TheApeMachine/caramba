package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/agent"
	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/utils"
)

type Break struct {
	Tool string `json:"tool" jsonschema:"title=Tool,description=The tool to break out of your iteration loop,enum=break,required"`
	Args struct {
		FinalAnswer string `json:"final_answer" jsonschema:"title=Final Answer,description=Your final answer, synthesized from your iterations,required"`
	} `json:"args" jsonschema:"title=Arguments,description=The arguments to pass to the tool,required"`
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
	return "Break out of your iteration loop"
}

func (brk *Break) GenerateSchema() any {
	return utils.GenerateSchema[Break]()
}

func (brk *Break) Use(
	accumulator *stream.Accumulator,
	input map[string]any,
	generators ...*agent.Generator,
) *stream.Accumulator {
	for _, generator := range generators {
		generator.Status = agent.AgentStatusIdle

		accumulator.Append(
			utils.QuickWrap("BREAK", utils.JoinWith("\n",
				"NAME  : "+generator.Ctx.Config.Name,
				"STATUS: IDLE",
			), 1),
		)
	}

	return accumulator
}

func (brk *Break) Connect(ctx context.Context, rwc io.ReadWriteCloser) error {
	return nil
}
