package xla

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Runner implements the runner.Runner interface for XLA execution.
*/
type Runner struct {
	platform string
	backend  *TensorBackend
	err      error
}

/*
NewRunner instantiates a new XLA runner.
*/
func NewRunner() *Runner {
	return NewRunnerForPlatform("cpu")
}

func NewRunnerForPlatform(platform string) *Runner {
	backend, err := NewTensorBackend(platform)

	return &Runner{
		platform: platform,
		backend:  backend,
		err:      err,
	}
}

/*
Execute traverses the intermediate representation graph and executes operations on XLA.
*/
func (runner *Runner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if graph == nil {
		return nil, fmt.Errorf("xla runner: graph is required")
	}

	if len(targets) == 0 {
		return nil, fmt.Errorf("xla runner: no execution targets provided")
	}

	if runner.err != nil {
		return nil, runner.err
	}

	nodes, tensors, err := runner.specs(graph, targets)

	if err != nil {
		return nil, err
	}

	outputs, err := executor.New(runner.backend).Execute(ctx, nodes, tensors)

	if err != nil {
		return nil, err
	}

	return outputs, nil
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
	if runner.backend == nil {
		return nil
	}

	return runner.backend.Close()
}

func (runner *Runner) specs(
	graph *ir.Graph, targets []*ir.Node,
) ([]executor.NodeSpec, []executor.TensorSpec, error) {
	targetsByID := targetSet(targets)
	nodes := graph.Nodes()
	nodeSpecs := make([]executor.NodeSpec, len(nodes))
	tensorSpecs := make([]executor.TensorSpec, 0)

	for index, node := range nodes {
		inputs := node.Inputs()
		inputIDs := make([]string, len(inputs))

		for inputIndex, input := range inputs {
			inputIDs[inputIndex] = input.ID()
		}

		metadata := node.Metadata()
		nodeSpecs[index] = executor.NodeSpec{
			ID:       node.ID(),
			Op:       node.OpType(),
			Inputs:   inputIDs,
			Shape:    node.Shape().Dims(),
			Metadata: metadata,
			Target:   targetsByID[node.ID()],
		}

		if node.OpType() != ir.OpInput {
			continue
		}

		tensorSpec, err := inputTensorSpec(node, metadata)

		if err != nil {
			return nil, nil, err
		}

		tensorSpecs = append(tensorSpecs, tensorSpec)
	}

	return nodeSpecs, tensorSpecs, nil
}

func inputTensorSpec(node *ir.Node, metadata map[string]any) (executor.TensorSpec, error) {
	rawValues, ok := metadata["values"]

	if !ok {
		return executor.TensorSpec{}, fmt.Errorf("xla runner: input node %s has no values metadata", node.ID())
	}

	values, ok := rawValues.([]float64)

	if !ok {
		return executor.TensorSpec{}, fmt.Errorf("xla runner: input node %s values metadata must be []float64", node.ID())
	}

	data, err := executor.EncodeFloat64(values)

	if err != nil {
		return executor.TensorSpec{}, err
	}

	dimensions := node.Shape().Dims()
	shape := make([]int64, len(dimensions))

	for index, dimension := range dimensions {
		shape[index] = int64(dimension)
	}

	return executor.TensorSpec{
		ID:    node.ID(),
		Shape: shape,
		Data:  data,
		DType: tensor.Float64,
	}, nil
}

func targetSet(targets []*ir.Node) map[string]bool {
	set := make(map[string]bool, len(targets))

	for _, target := range targets {
		set[target.ID()] = true
	}

	return set
}
