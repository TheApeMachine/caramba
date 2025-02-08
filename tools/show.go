package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

type Show struct {
	Tool   string `json:"tool" jsonschema:"title=Tool,description=The tool to show information about the system,enum=show,required"`
	Agents string `json:"agents" jsonschema:"title=Agents,description=Set to any value to show agents"`
}

func NewShow() *Show {
	return &Show{
		Tool: "show",
	}
}

func (s *Show) Name() string {
	return "show"
}

func (s *Show) Description() string {
	return "Show information about the system state"
}

func (s *Show) GenerateSchema() interface{} {
	return utils.GenerateSchema[Show]()
}

func (s *Show) Use(input map[string]any) string {
	return "show"
}

func (s *Show) Connect(ctx context.Context, rwc io.ReadWriteCloser) error {
	return nil
}
