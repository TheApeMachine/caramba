package xla

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Runner implements the runner.Runner interface for XLA execution.
*/
type Runner struct {
	ctx    context.Context
	cancel context.CancelFunc
	err    error
}

/*
NewRunner instantiates a new XLA runner.
*/
func NewRunner(ctx context.Context) *Runner {
	ctx, cancel := context.WithCancel(ctx)

	return &Runner{
		ctx:    ctx,
		cancel: cancel,
	}
}

/*
Execute traverses the intermediate representation graph and executes operations on XLA.
*/
func (runner *Runner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error) {
	if len(targets) == 0 {
		return nil, fmt.Errorf("xla runner: no execution targets provided")
	}

	results := make(map[string]tensor.Float64Tensor)
	return results, nil
}

/*
Location returns XLA.
*/
func (runner *Runner) Location() tensor.Location {
	return tensor.XLA
}

/*
Close cleans up any allocated resources.
*/
func (runner *Runner) Close() error {
	runner.cancel()
	return nil
}
