package runner

import (
	"context"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Runner represents a physical execution backend (CPU, CUDA, Metal, XLA).
It maps intermediate representation graphs into hardware-specific calls.
*/
type Runner interface {
	/*
		Execute evaluates the computation graph for the specified output nodes.
		It returns a map of node IDs to their computed float64 tensors.
	*/
	Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error)

	/*
		Location returns the location string denoting the runner's hardware context.
	*/
	Location() tensor.Location

	/*
		Close releases any hardware resources allocated by the runner.
	*/
	Close() error
}
