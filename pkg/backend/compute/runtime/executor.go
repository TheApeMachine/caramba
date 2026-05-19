package runtime

import (
	"context"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Executor evaluates lowered IR on one device. Each Location (host, metal,
cuda, xla) supplies its own implementation; Backend routes graphs here.
*/
type Executor interface {
	Execute(
		ctx context.Context,
		graph *ir.Graph,
		targets []*ir.Node,
	) (map[string]tensor.Tensor, error)

	Location() tensor.Location

	Close() error
}
