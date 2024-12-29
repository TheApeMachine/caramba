package ai

import (
	"context"
	"io"

	"github.com/theapemachine/caramba/tools"
)

type Tool interface {
	GenerateSchema() interface{}
	Use(context.Context, map[string]any) string
	Connect(context.Context, io.ReadWriteCloser) error
}

func NewToolset(role string) []Tool {
	return []Tool{
		toolMap[role],
	}
}

var toolsets = map[string][]Tool{
	"manager": {
		tools.NewAzure(),
	},
	"researcher": {
		tools.NewBrowser(),
	},
	"developer": {
		tools.NewContainer(),
	},
}

var toolMap = map[string]Tool{
	"browser": tools.NewBrowser(),
	"azure":   tools.NewAzure(),
}
