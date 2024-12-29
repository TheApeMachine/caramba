package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

type Trengo struct {
	Operation string `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=search,enum=add_labels,enum=get_labels,required"`
}

func (trengo *Trengo) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Trengo]()
}

func (trengo *Trengo) Use(ctx context.Context, params map[string]any) string {
	return ""
}

func (trengo *Trengo) Connect(rw io.ReadWriteCloser) error {
	return nil
}
