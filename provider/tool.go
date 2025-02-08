package provider

import (
	"context"
	"io"
)

type Tool interface {
	Name() string
	Description() string
	GenerateSchema() interface{}
	Use(context.Context, map[string]any) (string, error)
	Connect(context.Context, io.ReadWriteCloser) error
}
