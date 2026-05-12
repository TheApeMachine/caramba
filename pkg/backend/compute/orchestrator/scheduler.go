package orchestrator

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/runner"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Scheduler handles the routing and execution of intermediate representation graphs.
It accepts a computational graph, performs optimizations, and delegates execution
to the appropriate hardware runner.
*/
type Scheduler struct {
	ctx     context.Context
	cancel  context.CancelFunc
	err     error
	runners map[tensor.Location]runner.Runner
	fusion  *FusionOptimizer
	cse     *CSEOptimizer
	dce     *DCEOptimizer
}

/*
NewScheduler instantiates a new execution scheduler.
*/
func NewScheduler(ctx context.Context) *Scheduler {
	ctx, cancel := context.WithCancel(ctx)

	return &Scheduler{
		ctx:     ctx,
		cancel:  cancel,
		runners: make(map[tensor.Location]runner.Runner),
		fusion:  NewFusionOptimizer(ctx),
		cse:     NewCSEOptimizer(ctx),
		dce:     NewDCEOptimizer(ctx),
	}
}

/*
RegisterRunner adds a backend implementation for a specific hardware target.
*/
func (scheduler *Scheduler) RegisterRunner(r runner.Runner) {
	scheduler.runners[r.Location()] = r
}

/*
Execute optimizes the graph and dispatches it to the runner matching the requested location.
*/
func (scheduler *Scheduler) Execute(
	graph *ir.Graph,
	targets []*ir.Node,
	location tensor.Location,
) (map[string]tensor.Float64Tensor, error) {
	r, ok := scheduler.runners[location]

	if !ok {
		return nil, fmt.Errorf("scheduler: no runner registered for location %q", location)
	}

	optimizedGraph := scheduler.cse.Optimize(graph)
	optimizedGraph = scheduler.fusion.Optimize(optimizedGraph)

	if len(targets) == 0 {
		targets = optimizedGraph.Sinks()
	}

	optimizedGraph = scheduler.dce.Optimize(optimizedGraph, targets)

	// Future work: Track target ID mapping across fusions.
	// For now, always return the graph sinks.
	return r.Execute(scheduler.ctx, optimizedGraph, optimizedGraph.Sinks())
}
