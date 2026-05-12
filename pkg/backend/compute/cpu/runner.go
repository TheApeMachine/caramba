package cpu

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Runner implements the runner.Runner interface for CPU execution.
*/
type Runner struct {
	ctx    context.Context
	cancel context.CancelFunc
	err    error
}

/*
NewRunner instantiates a new CPU runner.
*/
func NewRunner(ctx context.Context) *Runner {
	ctx, cancel := context.WithCancel(ctx)

	return &Runner{
		ctx:    ctx,
		cancel: cancel,
	}
}

/*
Execute traverses the intermediate representation graph and executes operations on CPU.
*/
func (runner *Runner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error) {
	// Abstract dispatch logic mapping ir.Node to CPU specific kernel execution.
	// Currently a stub to fulfill architectural boundary constraints.
	if len(targets) == 0 {
		return nil, fmt.Errorf("cpu runner: no execution targets provided")
	}

	results := make(map[string]tensor.Float64Tensor)
	return results, nil
}

/*
Location returns Host.
*/
func (runner *Runner) Location() tensor.Location {
	return tensor.Host
}

/*
Close cleans up any allocated resources.
*/
func (runner *Runner) Close() error {
	runner.cancel()
	return nil
}
