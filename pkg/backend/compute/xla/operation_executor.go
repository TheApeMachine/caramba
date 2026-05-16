//go:build cgo && xla

package xla

import (
	"context"
	"fmt"
	"strings"

	"github.com/theapemachine/caramba/pkg/backend/compute/dispatch"
	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

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
		return nil, fmt.Errorf("xla tensor: input node %q was not materialized", node.ID)
	case ir.OpAdd:
		return requireXLAInputs(node, inputs, 2, tensorBackend.Add)
	case ir.OpMul:
		return requireXLAInputs(node, inputs, 2, tensorBackend.Mul)
	case ir.OpMatmul:
		return requireXLAInputs(node, inputs, 2, tensorBackend.Matmul)
	case ir.OpReLU:
		return requireXLAInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.ReLU(input)
		})
	case ir.OpLeakyReLU:
		return requireXLAInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.LeakyReLU(input, 0.01)
		})
	case ir.OpGELU:
		return requireXLAInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.GELU(input)
		})
	case ir.OpTanh:
		return requireXLAInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.Tanh(input)
		})
	case ir.OpSigmoid:
		return requireXLAInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.Sigmoid(input)
		})
	case ir.OpSwish:
		return requireXLAInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.Swish(input)
		})
	case ir.OpSELU:
		return requireXLAInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.SELU(input)
		})
	case ir.OpSwiGLU:
		return requireXLAInputs(node, inputs, 1, func(
			input, _ tensor.Float64Tensor,
		) (tensor.Float64Tensor, error) {
			return tensorBackend.SwiGLU(input)
		})
	case ir.OpFused:
		if len(inputs) != 3 {
			return nil, fmt.Errorf("xla tensor: Fused node %q requires 3 inputs", node.ID)
		}

		activation, _ := node.Metadata["activation"].(string)
		if strings.EqualFold(activation, string(ir.OpGELU)) {
			return tensorBackend.MatmulAddGELU(inputs[0], inputs[1], inputs[2])
		}

		return tensorBackend.MatmulAdd(inputs[0], inputs[1], inputs[2])
	default:
		return dispatch.RunOperation(
			ctx,
			tensorBackend,
			node,
			inputs,
			NewOperationRegistry(),
			NewOptimizerRegistryForPlatform(tensorBackend.platform),
		)
	}
}

func requireXLAInputs(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
	count int,
	apply func(tensor.Float64Tensor, tensor.Float64Tensor) (tensor.Float64Tensor, error),
) (tensor.Float64Tensor, error) {
	if len(inputs) != count {
		return nil, fmt.Errorf("xla tensor: %s node %q requires %d inputs", node.Op, node.ID, count)
	}

	var second tensor.Float64Tensor
	if len(inputs) > 1 {
		second = inputs[1]
	}

	return apply(inputs[0], second)
}
