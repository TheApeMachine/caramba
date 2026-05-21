package runtime

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

/*
Host executes IR graphs against tensor.Host resident storage.
*/
type Host struct {
	memory tensor.Backend
}

func NewHost(memory tensor.Backend) *Host {
	return &Host{memory: memory}
}

func (host *Host) Execute(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
) (map[string]tensor.Tensor, error) {
	return host.ExecuteWithWeights(ctx, graph, targets, "", nil)
}

/*
ExecuteWithWeights runs a graph using optional checkpoint bytes and external inputs.
*/
func (host *Host) ExecuteWithWeights(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
	weightsPath string,
	externalInputs map[string]tensor.Tensor,
) (map[string]tensor.Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if graph == nil {
		return nil, fmt.Errorf("host executor: graph is required")
	}

	if len(targets) == 0 {
		return nil, fmt.Errorf("host executor: no execution targets provided")
	}

	if err := validateTargets(graph, targets); err != nil {
		return nil, err
	}

	runner := NewGraphRunner(host.memory)

	return runner.Execute(ctx, graph, targets, weightsPath, externalInputs)
}

func (host *Host) Location() tensor.Location {
	return tensor.Host
}

func (host *Host) Close() error {
	if host == nil || host.memory == nil {
		return nil
	}

	return host.memory.Close()
}

var (
	_ Executor         = (*Host)(nil)
	_ WeightedExecutor = (*Host)(nil)
)

func validateTargets(graph *ir.Graph, targets []*ir.Node) error {
	index, err := graph.Index()

	if err != nil {
		return err
	}

	for _, target := range targets {
		if target == nil {
			return fmt.Errorf("host executor: nil execution target")
		}

		if index.Node(target.ID()) == nil {
			return fmt.Errorf("host executor: target node %q is not registered in graph", target.ID())
		}
	}

	return nil
}

func closeValues(values map[string]tensor.Tensor) {
	for _, value := range values {
		if value != nil {
			_ = value.Close()
		}
	}
}
