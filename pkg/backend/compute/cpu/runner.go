package cpu

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
Runner implements the runner.Runner interface for CPU execution.
*/
type Runner struct {
	backend *TensorBackend
}

/*
NewRunner instantiates a new CPU runner.
*/
func NewRunner() *Runner {
	return &Runner{
		backend: NewTensorBackend(),
	}
}

/*
Execute traverses the intermediate representation graph and executes operations on CPU.
*/
func (runner *Runner) Execute(ctx context.Context, graph *ir.Graph, targets []*ir.Node) (map[string]tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(targets) == 0 {
		return nil, fmt.Errorf("cpu runner: no execution targets provided")
	}

	nodes, tensors, err := runner.specs(graph, targets)

	if err != nil {
		return nil, err
	}

	outputs, err := executor.New(runner.backend).Execute(ctx, nodes, tensors)

	if err != nil {
		return nil, err
	}

	return runner.results(outputs)
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
		return executor.TensorSpec{}, fmt.Errorf("cpu runner: input node %s has no values metadata", node.ID())
	}

	values, ok := rawValues.([]float64)

	if !ok {
		return executor.TensorSpec{}, fmt.Errorf("cpu runner: input node %s values metadata must be []float64", node.ID())
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

func (runner *Runner) results(
	outputs []executor.TensorSpec,
) (map[string]tensor.Float64Tensor, error) {
	results := make(map[string]tensor.Float64Tensor, len(outputs))

	for _, output := range outputs {
		if output.DType != tensor.Float64 {
			return nil, fmt.Errorf("cpu runner: unsupported output dtype %q", output.DType)
		}

		values, err := executor.DecodeFloat64(output.Data)

		if err != nil {
			return nil, err
		}

		shape, err := tensorShape(output.Shape)

		if err != nil {
			return nil, err
		}

		uploaded, err := runner.backend.UploadFloat64(shape, values)

		if err != nil {
			return nil, err
		}

		results[output.ID] = uploaded
	}

	return results, nil
}

func tensorShape(shape []int64) (tensor.Shape, error) {
	dimensions := make([]int, len(shape))

	for index, dimension := range shape {
		dimensions[index] = int(dimension)
	}

	return tensor.NewShape(dimensions)
}

func targetSet(targets []*ir.Node) map[string]bool {
	set := make(map[string]bool, len(targets))

	for _, target := range targets {
		set[target.ID()] = true
	}

	return set
}
