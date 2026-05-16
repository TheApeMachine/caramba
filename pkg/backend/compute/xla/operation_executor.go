//go:build cgo && xla

package xla

import (
	"context"
	"fmt"
	"strings"

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
		return tensorBackend.applyModelOperation(ctx, node, inputs)
	}
}

func (tensorBackend *TensorBackend) applyModelOperation(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	switch strings.ToLower(string(node.Op)) {
	case "shape.reshape":
		return tensorBackend.applyReshape(node, inputs)
	case "shape.transpose":
		return tensorBackend.applyTranspose(node, inputs)
	case "shape.concat":
		return tensorBackend.applyConcat(node, inputs)
	case "shape.split":
		return tensorBackend.applySplit(node, inputs)
	case "shape.upsample_nearest2d":
		return tensorBackend.applyUpsampleNearest2D(node, inputs)
	case "shape.view_as_heads":
		return tensorBackend.applyViewAsHeads(node, inputs)
	case "shape.merge_heads":
		return tensorBackend.applyMergeHeads(node, inputs)
	case "shape.last_token":
		return tensorBackend.applyLastToken(node, inputs)
	default:
		return nil, fmt.Errorf(
			"xla tensor: operation %q node %q has no resident XLA implementation",
			node.Op,
			node.ID,
		)
	}
}

func (tensorBackend *TensorBackend) applyReshape(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("xla tensor: reshape node %q requires 1 input", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	return tensorBackend.ReshapeTensor(inputs[0], outputShape)
}

func (tensorBackend *TensorBackend) applyTranspose(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("xla tensor: transpose node %q requires 1 input", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	return tensorBackend.TransposeTensor(
		inputs[0],
		outputShape,
		intConfig(node, "dim0", 0),
		intConfig(node, "dim1", 1),
	)
}

func (tensorBackend *TensorBackend) applyConcat(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if len(inputs) < 2 {
		return nil, fmt.Errorf("xla tensor: concat node %q requires at least 2 inputs", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	if len(inputs) == 2 {
		return tensorBackend.ConcatTensor(inputs[0], inputs[1], outputShape)
	}

	current := inputs[0]
	var temporary tensor.Float64Tensor

	for inputIndex := 1; inputIndex < len(inputs); inputIndex++ {
		nextLength := current.Shape().Len() + inputs[inputIndex].Shape().Len()
		nextShape := outputShape

		if inputIndex != len(inputs)-1 {
			nextShape, err = tensor.NewShape([]int{nextLength})

			if err != nil {
				if temporary != nil {
					_ = temporary.Close()
				}

				return nil, err
			}
		}

		next, err := tensorBackend.ConcatTensor(current, inputs[inputIndex], nextShape)

		if temporary != nil {
			_ = temporary.Close()
		}

		if err != nil {
			return nil, err
		}

		current = next
		temporary = next
	}

	return current, nil
}

func (tensorBackend *TensorBackend) applySplit(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("xla tensor: split node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()
	dimension := intConfig(node, "dim", 0)

	if dimension < 0 || dimension >= len(inputShape) {
		return nil, fmt.Errorf("xla tensor: split node %q dimension out of range", node.ID)
	}

	outer, inner, err := splitOuterInner(inputShape, dimension)

	if err != nil {
		return nil, err
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	return tensorBackend.SplitTensor(
		inputs[0],
		outputShape,
		outer,
		inputShape[dimension],
		intConfig(node, "split_size", inputShape[dimension]),
		inner,
	)
}

func (tensorBackend *TensorBackend) applyUpsampleNearest2D(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("xla tensor: upsample_nearest2d node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("xla tensor: upsample_nearest2d node %q expects rank 4", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	outputDims := outputShape.Dims()

	if len(outputDims) != 4 {
		return nil, fmt.Errorf("xla tensor: upsample_nearest2d node %q output must be rank 4", node.ID)
	}

	scaleH := intConfig(node, "scale_factor", 0)
	scaleW := intConfig(node, "scale_factor", 0)
	scaleH = intConfig(node, "scale_h", scaleH)
	scaleW = intConfig(node, "scale_w", scaleW)

	if scaleH == 0 && outputDims[2] > 0 && outputDims[2]%inputShape[2] == 0 {
		scaleH = outputDims[2] / inputShape[2]
	}

	if scaleW == 0 && outputDims[3] > 0 && outputDims[3]%inputShape[3] == 0 {
		scaleW = outputDims[3] / inputShape[3]
	}

	return tensorBackend.UpsampleNearest2DTensor(
		inputs[0],
		outputShape,
		inputShape[0],
		inputShape[1],
		inputShape[2],
		inputShape[3],
		scaleH,
		scaleW,
	)
}

func (tensorBackend *TensorBackend) applyViewAsHeads(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("xla tensor: view_as_heads node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) != 3 {
		return nil, fmt.Errorf("xla tensor: view_as_heads node %q expects rank 3", node.ID)
	}

	numHeads := intConfig(node, "num_heads", 0)

	if numHeads <= 0 || inputShape[2]%numHeads != 0 {
		return nil, fmt.Errorf("xla tensor: invalid head count %d", numHeads)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	return tensorBackend.ViewAsHeadsTensor(
		inputs[0],
		outputShape,
		inputShape[0],
		inputShape[1],
		numHeads,
		inputShape[2]/numHeads,
	)
}

func (tensorBackend *TensorBackend) applyMergeHeads(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("xla tensor: merge_heads node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("xla tensor: merge_heads node %q expects rank 4", node.ID)
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	return tensorBackend.MergeHeadsTensor(
		inputs[0],
		outputShape,
		inputShape[0],
		inputShape[1],
		inputShape[2],
		inputShape[3],
	)
}

func (tensorBackend *TensorBackend) applyLastToken(
	node executor.NodeSpec,
	inputs []tensor.Float64Tensor,
) (tensor.Float64Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("xla tensor: last_token node %q requires 1 input", node.ID)
	}

	inputShape := inputs[0].Shape().Dims()

	if len(inputShape) < 2 {
		return nil, fmt.Errorf("xla tensor: last_token node %q expects rank >= 2", node.ID)
	}

	outerLength, err := lastTokenOuterLength(inputShape)

	if err != nil {
		return nil, err
	}

	outputShape, err := tensor.NewShape(node.Shape)

	if err != nil {
		return nil, err
	}

	return tensorBackend.LastTokenTensor(
		inputs[0],
		outputShape,
		outerLength,
		inputShape[len(inputShape)-2],
		inputShape[len(inputShape)-1],
	)
}

func intConfig(node executor.NodeSpec, key string, fallback int) int {
	value, ok := node.Metadata[key]

	if !ok {
		return fallback
	}

	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case float32:
		return int(typed)
	default:
		return fallback
	}
}

func splitOuterInner(shape []int, dimension int) (int, int, error) {
	outer := 1

	for _, value := range shape[:dimension] {
		if value <= 0 {
			return 0, 0, fmt.Errorf("xla tensor: split outer dimensions must be positive")
		}

		outer *= value
	}

	inner := 1

	for _, value := range shape[dimension+1:] {
		if value <= 0 {
			return 0, 0, fmt.Errorf("xla tensor: split inner dimensions must be positive")
		}

		inner *= value
	}

	return outer, inner, nil
}

func lastTokenOuterLength(shape []int) (int, error) {
	outer := 1

	for _, value := range shape[:len(shape)-2] {
		if value <= 0 {
			return 0, fmt.Errorf("xla tensor: last_token outer dimensions must be positive")
		}

		outer *= value
	}

	return outer, nil
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
