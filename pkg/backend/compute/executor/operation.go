package executor

import (
	"context"
	"fmt"
	"strings"

	cpuoperation "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func RunOperation(
	ctx context.Context,
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	operation cpuoperation.Operation,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	values, err := InputValues(inputs)
	if err != nil {
		return nil, err
	}

	output := operation.Forward(OutputShape(node, inputs), values...)

	return UploadOutput(backend, node, inputs, output)
}

func RunErrorOperation(
	ctx context.Context,
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	operation func(shape []int, data ...[]float64) ([]float64, error),
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	values, err := InputValues(inputs)
	if err != nil {
		return nil, err
	}

	output, err := operation(OutputShape(node, inputs), values...)
	if err != nil {
		return nil, err
	}

	return UploadOutput(backend, node, inputs, output)
}

func RunForwardErrorOperation(
	ctx context.Context,
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	operation interface {
		Forward(shape []int, data ...[]float64) ([]float64, error)
	},
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	values, err := InputValues(inputs)
	if err != nil {
		return nil, err
	}

	output, err := operation.Forward(OutputShape(node, inputs), values...)
	if err != nil {
		return nil, err
	}

	return UploadOutput(backend, node, inputs, output)
}

func UploadOutput(
	backend Backend,
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	output []float64,
) (tensor.Float64Tensor, error) {
	shape, err := OutputTensorShape(node, inputs, output)

	if err != nil {
		return nil, err
	}

	return backend.UploadFloat64(shape, output)
}

func InputValues(inputs []tensor.Float64Tensor) ([][]float64, error) {
	values := make([][]float64, len(inputs))

	for index, input := range inputs {
		value, err := input.CloneFloat64()

		if err != nil {
			return nil, err
		}

		values[index] = value
	}

	return values, nil
}

func OutputShape(node NodeSpec, inputs []tensor.Float64Tensor) []int {
	if len(node.Shape) > 0 {
		return append([]int(nil), node.Shape...)
	}

	if len(inputs) == 0 {
		return nil
	}

	return inputs[0].Shape().Dims()
}

func OutputTensorShape(
	node NodeSpec,
	inputs []tensor.Float64Tensor,
	output []float64,
) (tensor.Shape, error) {
	shapeData := OutputShape(node, inputs)
	if len(shapeData) == 0 {
		shapeData = []int{len(output)}
	}

	shape, err := tensor.NewShape(shapeData)

	if err != nil {
		return tensor.Shape{}, err
	}

	if shape.Len() != len(output) {
		return tensor.Shape{}, fmt.Errorf(
			"executor: %s node %q output has %d values for shape length %d",
			node.Op,
			node.ID,
			len(output),
			shape.Len(),
		)
	}

	return shape, nil
}

func NormalizeOperation(op ir.OpType) ir.OpType {
	switch strings.ToLower(string(op)) {
	case "input", "data.input":
		return ir.OpInput
	case "add", "math.add":
		return ir.OpAdd
	case "mul", "math.mul":
		return ir.OpMul
	case "matmul", "math.matmul":
		return ir.OpMatmul
	case "relu", "activation.relu":
		return ir.OpReLU
	case "leakyrelu", "leaky_relu", "activation.leaky_relu":
		return ir.OpLeakyReLU
	case "gelu", "activation.gelu":
		return ir.OpGELU
	case "tanh", "activation.tanh":
		return ir.OpTanh
	case "sigmoid", "activation.sigmoid":
		return ir.OpSigmoid
	case "swiglu", "activation.swiglu":
		return ir.OpSwiGLU
	case "fused", "math.matmul_add", "math.matmul_add_gelu":
		return ir.OpFused
	default:
		return op
	}
}
