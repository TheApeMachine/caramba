package provider

import (
	"github.com/mark3labs/mcp-go/mcp"
	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/params"
	"github.com/theapemachine/caramba/pkg/datura"
)

type ProviderType interface {
	Generate(params.Params, aicontext.Context, []mcp.Tool) chan *datura.Artifact
}
