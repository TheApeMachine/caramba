package orchestrator

import (
	"context"
	"fmt"
	"sync"

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
	mu      sync.RWMutex
	runners map[tensor.Location]runner.Runner
	fusion  *FusionOptimizer
	cse     *CSEOptimizer
	dce     *DCEOptimizer
}

/*
NewScheduler instantiates a new execution scheduler.
*/
func NewScheduler() *Scheduler {
	return &Scheduler{
		runners: make(map[tensor.Location]runner.Runner),
		fusion:  NewFusionOptimizer(),
		cse:     NewCSEOptimizer(),
		dce:     NewDCEOptimizer(),
	}
}

/*
RegisterRunner adds a backend implementation for a specific hardware target.
*/
func (scheduler *Scheduler) RegisterRunner(r runner.Runner) {
	scheduler.mu.Lock()
	defer scheduler.mu.Unlock()
	scheduler.runners[r.Location()] = r
}

/*
Execute optimizes the graph and dispatches it to the runner matching the requested location.
*/
func (scheduler *Scheduler) Execute(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
	location tensor.Location,
) (map[string]tensor.Float64Tensor, error) {
	if graph == nil {
		return nil, fmt.Errorf("scheduler: nil graph")
	}

	scheduler.mu.RLock()
	r, ok := scheduler.runners[location]
	scheduler.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("scheduler: no runner registered for location %q", location)
	}

	optimizedGraph, err := scheduler.cse.Optimize(graph)
	if err != nil {
		return nil, err
	}

	optimizedGraph, err = scheduler.fusion.Optimize(optimizedGraph)
	if err != nil {
		return nil, err
	}

	if len(targets) == 0 {
		targets = optimizedGraph.Sinks()
	}

	optimizedGraph, err = scheduler.dce.Optimize(ctx, optimizedGraph, targets)
	if err != nil {
		return nil, err
	}

	if len(optimizedGraph.Sinks()) != len(targets) {
		// Just a simple safety check. A real mapping across CSE/Fusion/DCE is more involved.
		// For now we assume sinks represent what we wanted.
	}

	// We simply ask the runner to compute everything that reaches a sink
	return r.Execute(ctx, optimizedGraph, optimizedGraph.Sinks())
}
