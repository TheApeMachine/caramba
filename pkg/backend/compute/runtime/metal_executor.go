package runtime

import (
	"context"
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal"
)

/*
Metal executes IR graphs on the Metal device through direct kernel dispatch.
*/
type Metal struct {
	memory *metal.Backend
}

/*
NewMetal constructs a Metal executor backed by a Metal memory backend.
*/
func NewMetal(memory *metal.Backend) *Metal {
	return &Metal{memory: memory}
}

func (metalExecutor *Metal) Execute(
	ctx context.Context,
	graph *ir.Graph,
	targets []*ir.Node,
) (map[string]tensor.Tensor, error) {
	return metalExecutor.ExecuteWithWeights(ctx, graph, targets, "", nil)
}

/*
ExecuteWithWeights runs a graph using optional checkpoint bytes and external inputs.
*/
func (metalExecutor *Metal) ExecuteWithWeights(
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
		return nil, fmt.Errorf("metal executor: graph is required")
	}

	if len(targets) == 0 {
		return nil, fmt.Errorf("metal executor: no execution targets provided")
	}

	if err := validateTargets(graph, targets); err != nil {
		return nil, err
	}

	runner := NewMetalGraphRunner(metalExecutor.memory)

	return runner.Execute(ctx, graph, targets, weightsPath, externalInputs)
}

func (metalExecutor *Metal) Location() tensor.Location {
	return tensor.Metal
}

func (metalExecutor *Metal) Close() error {
	return nil
}

var (
	_ Executor         = (*Metal)(nil)
	_ WeightedExecutor = (*Metal)(nil)
)
