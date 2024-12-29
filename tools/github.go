package tools

import (
	"context"

	"github.com/theapemachine/caramba/utils"
)

type Github struct {
	Repo      string `json:"repository" jsonschema:"title=Repository,description=The repository to use,required"`
	Operation string `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=clone,enum=pull,enum=push,required"`
}

func (g *Github) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Github]()
}

func (github *Github) Use(ctx context.Context, params map[string]any) string {
	return ""
}
