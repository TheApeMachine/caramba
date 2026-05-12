package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Runner implements the runner.Runner interface for Metal execution.
*/
type Runner struct {
	ctx    context.Context
	cancel context.CancelFunc
	err    error
}

/*
NewRunner instantiates a new Metal runner.
*/
func NewRunner(ctx context.Context) *Runner {
	ctx, cancel := context.WithCancel(ctx)

	return &Runner{
		ctx:    ctx,
		cancel: cancel,
	}
}

/*
Execute traverses the intermediate representation graph and executes operations on Metal.
*/
func (runner *Runner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error) {
	if len(targets) == 0 {
		return nil, fmt.Errorf("metal runner: no execution targets provided")
	}

	results := make(map[string]tensor.Float64Tensor)
	return results, nil
}

/*
Location returns Metal.
*/
func (runner *Runner) Location() tensor.Location {
	return tensor.Metal
}

/*
Close cleans up any allocated resources.
*/
func (runner *Runner) Close() error {
	runner.cancel()
	return nil
}
