//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

var _ executor.Backend = (*TensorBackend)(nil)

func (tensorBackend *TensorBackend) Apply(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	switch executor.NormalizeOperation(node.Op) {
	case ir.OpInput:
		return nil, fmt.Errorf("metal tensor: input node %q was not materialized", node.ID)
	case ir.OpAdd:
		return requireMetalInputs(node, inputs, 2, tensorBackend.Add)
	case ir.OpMul:
		return requireMetalInputs(node, inputs, 2, tensorBackend.Mul)
	case ir.OpMatmul:
		return requireMetalInputs(node, inputs, 2, tensorBackend.Matmul)
	case ir.OpReLU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.ReLU(input)
		})
	case ir.OpLeakyReLU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.LeakyReLU(input, 0.01)
		})
	case ir.OpGELU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.GELU(input)
		})
	case ir.OpTanh:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.Tanh(input)
		})
	case ir.OpSigmoid:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.Sigmoid(input)
		})
	case ir.OpSwiGLU:
		return requireMetalInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.SwiGLU(input)
		})
	case ir.OpFused:
		if len(inputs) != 3 {
			return nil, fmt.Errorf("metal tensor: Fused node %q requires 3 inputs", node.ID)
		}

		activation, _ := node.Metadata["activation"].(string)
		if strings.EqualFold(activation, string(ir.OpGELU)) {
			return tensorBackend.MatmulAddGELU(inputs[0], inputs[1], inputs[2])
		}

		return tensorBackend.MatmulAdd(inputs[0], inputs[1], inputs[2])
	default:
		return nil, fmt.Errorf("metal tensor: unsupported operation %q", node.Op)
	}
}

func (tensorBackend *TensorBackend) ReLU(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return (*MetalActivation)(nil).ReLUTensor(input)
}

func (tensorBackend *TensorBackend) LeakyReLU(
	input tensor.Float64Tensor, alpha float64,
) (tensor.Float64Tensor, error) {
	return (*MetalActivation)(nil).LeakyReLUTensor(input, alpha)
}

func (tensorBackend *TensorBackend) GELU(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return (*MetalActivation)(nil).GELUTensor(input)
}

func (tensorBackend *TensorBackend) Tanh(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return (*MetalActivation)(nil).TanhTensor(input)
}

func (tensorBackend *TensorBackend) Sigmoid(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return (*MetalActivation)(nil).SigmoidTensor(input)
}

func (tensorBackend *TensorBackend) SwiGLU(
	input tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return (*MetalActivation)(nil).SwiGLUTensor(input)
}

func (tensorBackend *TensorBackend) Add(
	left, right tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return (*MathOps)(nil).AddTensor(left, right)
}

func (tensorBackend *TensorBackend) Mul(
	left, right tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return (*MathOps)(nil).MulTensor(left, right)
}

func (tensorBackend *TensorBackend) Matmul(
	left, right tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return (*MathOps)(nil).MatmulTensor(left, right)
}

func (tensorBackend *TensorBackend) MatmulAdd(
	left, right, bias tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return (*MathOps)(nil).MatmulAddTensor(left, right, bias)
}

func (tensorBackend *TensorBackend) MatmulAddGELU(
	left, right, bias tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	return (*MathOps)(nil).MatmulAddGELUTensor(left, right, bias)
}

func requireMetalInputs(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
	count int,
	apply func(tensor.Float64Tensor, tensor.Float64Tensor) (tensor.Float64Tensor, error),
) (tensor.Float64Tensor, error) {
	if len(inputs) != count {
		return nil, fmt.Errorf("metal tensor: %s node %q requires %d inputs", node.Op, node.ID, count)
	}

	var second tensor.Float64Tensor
	if len(inputs) > 1 {
		second = inputs[1]
	}

	return apply(inputs[0], second)
}
