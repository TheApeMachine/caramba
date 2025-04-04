package provider

import (
	"context"

	aicontext "github.com/theapemachine/caramba/pkg/ai/context"
	"github.com/theapemachine/caramba/pkg/ai/params"
	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/tools"
)

type ProviderType interface {
	Generate(params.Params, aicontext.Context, []tools.Tool) chan *datura.Artifact
}

// GenerateServer implements the Generate interface from the Cap'n Proto definition
type GenerateServer struct {
	prvdr ProviderType
}

// Call implements the synchronous generation method
func (srv *GenerateServer) Call(
	ctx context.Context,
	p params.Params,
	c aicontext.Context,
	t []tools.Tool,
) chan *datura.Artifact {
	return srv.prvdr.Generate(p, c, t)
}

// Stream implements the streaming generation method
func (srv *GenerateServer) Stream(
	ctx context.Context,
	p params.Params,
	c aicontext.Context,
	t []tools.Tool,
) chan *datura.Artifact {
	return srv.prvdr.Generate(p, c, t)
}
