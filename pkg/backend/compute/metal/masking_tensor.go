//go:build darwin && cgo

package metal

// #include "masking.h"
import "C"

import (
	"fmt"
	"math"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
ApplyMaskTensor performs resident Metal additive masking.
*/
func (masking *MetalMasking) ApplyMaskTensor(
	scores, mask computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	metalScores, err := requireMetalTensor(scores)
	if err != nil {
		return nil, err
	}

	metalMask, err := requireMetalTensor(mask)
	if err != nil {
		return nil, err
	}

	if !metalScores.shape.Equal(metalMask.shape) {
		return nil, fmt.Errorf("metal tensor: masking.apply requires matching input shapes")
	}

	output, err := masking.runtime.NewFloat32Tensor(metalScores.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_apply_mask_tensor(
		metalScores.buffer,
		metalMask.buffer,
		output.buffer,
		C.int(metalScores.shape.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: masking.apply launch failed")
	}

	return output, nil
}

/*
CausalMaskTensor generates a resident Metal causal attention mask.
*/
func (masking *MetalMasking) CausalMaskTensor(
	outputShape computetensor.Shape,
	seqLen int,
) (computetensor.Float64Tensor, error) {
	if seqLen < 0 {
		return nil, fmt.Errorf("metal tensor: masking.causal sequence length must be non-negative")
	}

	if outputShape.Len() != seqLen*seqLen {
		return nil, fmt.Errorf("metal tensor: masking.causal output shape must contain seq_len^2 values")
	}

	output, err := masking.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_causal_mask_tensor(output.buffer, C.int(seqLen))
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: masking.causal launch failed")
	}

	return output, nil
}

func causalMaskSeqLen(outputShape computetensor.Shape, metadata map[string]any) (int, error) {
	if seqLen := intMetadata(metadata, "seq_len"); seqLen >= 0 {
		return seqLen, nil
	}

	dimensions := outputShape.Dims()
	if len(dimensions) >= 2 && dimensions[len(dimensions)-1] == dimensions[len(dimensions)-2] {
		return dimensions[len(dimensions)-1], nil
	}

	root := int(math.Sqrt(float64(outputShape.Len())))
	if root*root == outputShape.Len() {
		return root, nil
	}

	return 0, fmt.Errorf("metal tensor: masking.causal requires seq_len metadata or square output shape")
}

func intMetadata(metadata map[string]any, key string) int {
	value, ok := metadata[key]
	if !ok {
		return -1
	}

	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return -1
	}
}
