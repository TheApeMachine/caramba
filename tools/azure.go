package tools

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/utils"
)

type Azure struct {
	Operation string `json:"operation" jsonschema:"title=Operation,description=The operation to perform,enum=search,enum=create_ticket,enum=get_tickets,enum=update_ticket,required"`
}

func NewAzure() *Azure {
	return &Azure{}
}

func (azure *Azure) GenerateSchema() interface{} {
	return utils.GenerateSchema[*Azure]()
}

func (azure *Azure) Use(ctx context.Context, params map[string]any) string {
	return ""
}

func (azure *Azure) Connect(ctx context.Context, conn io.ReadWriteCloser) error {
	return nil
}
