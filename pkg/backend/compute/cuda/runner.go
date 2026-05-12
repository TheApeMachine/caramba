package cuda

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Runner implements the runner.Runner interface for CUDA execution.
*/
type Runner struct {
	backend *TensorBackend
}

/*
NewRunner instantiates a new CUDA runner.
*/
func NewRunner() *Runner {
	return &Runner{
		backend: NewTensorBackend(),
	}
}

/*
Execute traverses the intermediate representation graph and executes operations on CUDA.
*/
func (runner *Runner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(targets) == 0 {
		return nil, fmt.Errorf("cuda runner: no execution targets provided")
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
Location returns CUDA.
*/
func (runner *Runner) Location() tensor.Location {
	return tensor.CUDA
}

/*
Close cleans up any allocated resources.
*/
func (runner *Runner) Close() error {
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
		return executor.TensorSpec{}, fmt.Errorf("cuda runner: input node %s has no values metadata", node.ID())
	}

	values, ok := rawValues.([]float64)

	if !ok {
		return executor.TensorSpec{}, fmt.Errorf("cuda runner: input node %s values metadata must be []float64", node.ID())
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
