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
}

/*
NewRunner instantiates a new Metal runner.
*/
func NewRunner() *Runner {
	return &Runner{}
}

/*
Execute traverses the intermediate representation graph and executes operations on Metal.
*/
func (runner *Runner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if graph == nil {
		return nil, fmt.Errorf("metal runner: graph is required")
	}

	if len(targets) == 0 {
		return nil, fmt.Errorf("metal runner: no execution targets provided")
	}

	return nil, fmt.Errorf("metal runner: graph execution is not wired to the Metal tensor backend")
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
	return nil
}
