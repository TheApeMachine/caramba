package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/stream"
	"github.com/theapemachine/caramba/types"
	"github.com/theapemachine/caramba/utils"
)

type Show struct {
	Tool string `json:"tool" jsonschema:"title=Tool,description=The tool to show information about the system,enum=show,required"`
	Args struct {
		Scope string `json:"scope" jsonschema:"title=Scope,description=The scope of the information to show,enum=agents,required"`
	} `json:"args" jsonschema:"title=Arguments,description=The arguments to pass to the tool,required"`
}

func NewShow() *Show {
	return &Show{}
}

func (show *Show) Name() string {
	return "show"
}

func (show *Show) Description() string {
	return "Show information about the system state"
}

func (show *Show) GenerateSchema() interface{} {
	return utils.GenerateSchema[Show]()
}

func (show *Show) Use(
	accumulator *stream.Accumulator,
	input map[string]any,
	generators ...types.Generator,
) *stream.Accumulator {
	var names []string

	for _, generator := range generators {
		names = append(names, utils.JoinWith("\n",
			"NAME: "+generator.Ctx().Config().Name(),
			"ROLE: "+generator.Ctx().Config().Role(),
		))
	}

	accumulator.Append(
		utils.QuickWrap("AGENTS", utils.JoinWith("\n", names...), 2),
	)

	return accumulator
}

func (show *Show) Connect(ctx context.Context, rwc io.ReadWriteCloser) error {
	return nil
}
