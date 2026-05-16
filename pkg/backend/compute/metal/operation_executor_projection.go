//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyFusedQKV(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) < 1 || len(inputs) > 3 {
		return nil, fmt.Errorf("metal tensor: fused_qkv node %q requires 1 to 3 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	rows, inFeatures, outFeatures, err := fusedQKVDimensions(node, inputs[0], outputShape)
	if err != nil {
		return nil, err
	}

	weightTensor, err := tensorBackend.fusedQKVWeight(node, inputs, inFeatures, outFeatures)
	if err != nil {
		return nil, err
	}

	biasTensor, err := tensorBackend.fusedQKVBias(node, inputs, outFeatures)
	if err != nil {
		return nil, err
	}

	projectionOps, err := tensorBackend.projection()
	if err != nil {
		return nil, err
	}

	return projectionOps.FusedQKVTensor(
		inputs[0],
		weightTensor,
		biasTensor,
		outputShape,
		rows,
		inFeatures,
		outFeatures,
	)
}

func fusedQKVDimensions(
	node executor.NodeSpec,
	input computetensor.Float64Tensor,
	outputShape computetensor.Shape,
) (int, int, int, error) {
	inFeatures := intConfig(node, "d_in", 0)
	inputShape := input.Shape().Dims()

	if inFeatures == 0 && len(inputShape) > 0 {
		inFeatures = inputShape[len(inputShape)-1]
	}

	outFeatures := intConfig(node, "d_q", 0) +
		intConfig(node, "d_k", 0) +
		intConfig(node, "d_v", 0)

	if inFeatures <= 0 {
		return 0, 0, 0, fmt.Errorf("metal tensor: fused_qkv node %q requires d_in", node.ID)
	}

	if input.Shape().Len()%inFeatures != 0 {
		return 0, 0, 0, fmt.Errorf("metal tensor: fused_qkv input length is not divisible by d_in")
	}

	rows := input.Shape().Len() / inFeatures

	if outFeatures == 0 && rows > 0 {
		outFeatures = outputShape.Len() / rows
	}

	if rows <= 0 || outFeatures <= 0 {
		return 0, 0, 0, fmt.Errorf("metal tensor: fused_qkv dimensions must be positive")
	}

	if outputShape.Len() != rows*outFeatures {
		return 0, 0, 0, fmt.Errorf("metal tensor: fused_qkv output shape mismatch")
	}

	return rows, inFeatures, outFeatures, nil
}

func (tensorBackend *TensorBackend) fusedQKVWeight(
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
	inFeatures int,
	outFeatures int,
) (computetensor.Float64Tensor, error) {
	weight := floatSliceConfig(node, "weight")

	if len(weight) == 0 {
		if len(inputs) < 2 {
			return nil, fmt.Errorf("metal tensor: fused_qkv node %q requires weight", node.ID)
		}

		return inputs[1], nil
	}

	return tensorBackend.cachedTensor(
		node.ID+":weight",
		[]int{inFeatures, outFeatures},
		weight,
	)
}

func (tensorBackend *TensorBackend) fusedQKVBias(
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
	outFeatures int,
) (computetensor.Float64Tensor, error) {
	bias := floatSliceConfig(node, "bias")

	if len(bias) == 0 {
		if len(inputs) < 3 {
			return nil, nil
		}

		return inputs[2], nil
	}

	return tensorBackend.cachedTensor(node.ID+":bias", []int{outFeatures}, bias)
}
