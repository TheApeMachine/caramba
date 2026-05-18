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
		graph must not be nil. targets must not be nil or empty; callers will receive an error if so.
		targets must refer to nodes present in graph, and missing nodes produce an error.
		If circular dependencies are detected, an error is returned.
		Context cancellation/timeouts should be respected to abort execution promptly.
		Execute implementations should be safe for concurrent use on the same Runner instance.
		It returns a map of node IDs to their computed tensor outputs.
	*/
	Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Tensor, error)

	/*
		Location returns the location string denoting the runner's hardware context.
	*/
	Location() tensor.Location

	/*
		Close releases any hardware resources allocated by the runner.
		It is safe to call multiple times (idempotent).
		Subsequent calls to Execute (and other methods) will return an error after Close has been called.
	*/
	Close() error
}
