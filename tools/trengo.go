package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

type Trengo struct {
	Operation string `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=search,enum=add_labels,enum=get_labels,required"`
}

func NewTrengo() *Trengo {
	return &Trengo{}
}

func (trengo *Trengo) Name() string {
	return "trengo"
}

func (trengo *Trengo) Description() string {
	return "Interact with Trengo"
}

func (trengo *Trengo) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Trengo]()
}

func (trengo *Trengo) Use(ctx context.Context, params map[string]any) string {
	return ""
}

func (trengo *Trengo) Connect(ctx context.Context, rw io.ReadWriteCloser) error {
	return nil
}
