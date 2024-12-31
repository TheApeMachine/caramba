package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

type Github struct {
	Repo      string `json:"repository" jsonschema:"title=Repository,description=The repository to use,required"`
	Operation string `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=clone,enum=pull,enum=push,required"`
}

func NewGithub() *Github {
	return &Github{}
}

func (g *Github) Name() string {
	return "github"
}

func (g *Github) Description() string {
	return "Interact with GitHub repositories"
}

func (g *Github) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Github]()
}

func (github *Github) Use(ctx context.Context, params map[string]any) string {
	return ""
}

func (github *Github) Connect(ctx context.Context, conn io.ReadWriteCloser) error {
	return nil
}
