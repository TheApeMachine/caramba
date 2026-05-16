//go:build darwin && cgo

package metal

import (
	"context"
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/executor"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (tensorBackend *TensorBackend) applyMask(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, fmt.Errorf("metal tensor: masking.apply node %q requires 2 inputs", node.ID)
	}

	maskingOps, err := tensorBackend.masking()
	if err != nil {
		return nil, err
	}

	return maskingOps.ApplyMaskTensor(inputs[0], inputs[1])
}

func (tensorBackend *TensorBackend) applyCausalMask(
	ctx context.Context,
	node executor.NodeSpec,
	inputs []computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(inputs) != 0 {
		return nil, fmt.Errorf("metal tensor: masking.causal node %q requires 0 inputs", node.ID)
	}

	outputShape, err := computetensor.NewShape(node.Shape)
	if err != nil {
		return nil, err
	}

	seqLen, err := causalMaskSeqLen(outputShape, node.Metadata)
	if err != nil {
		return nil, err
	}

	maskingOps, err := tensorBackend.masking()
	if err != nil {
		return nil, err
	}

	return maskingOps.CausalMaskTensor(outputShape, seqLen)
}
