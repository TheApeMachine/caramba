//go:build darwin && cgo

package metal

// #include "markov_blanket.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
PartitionTensor extracts resident Markov blanket partitions from concatenated masks.
*/
func (op *MetalMarkovBlanket) PartitionTensor(
	state computetensor.Tensor,
	masks computetensor.Tensor,
	outputShape computetensor.Shape,
	sensoryCount int,
	activeCount int,
	internalCount int,
	externalCount int,
) (computetensor.Tensor, error) {
	stateTensor, maskTensor, err := markovTwo(state, masks)
	if err != nil {
		return nil, err
	}

	stateCount := stateTensor.Len()
	outputCount := sensoryCount + activeCount + internalCount + externalCount
	if stateCount <= 0 || maskTensor.Len() != 4*stateCount || outputShape.Len() != outputCount {
		return nil, fmt.Errorf("metal markov blanket: partition shape mismatch")
	}

	output, err := op.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_mb_partition_tensor(
		stateTensor.buffer,
		maskTensor.buffer,
		output.buffer,
		C.int(stateCount),
		C.int(sensoryCount),
		C.int(activeCount),
		C.int(internalCount),
		C.int(externalCount),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_mb_partition_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
FlowInternalTensor computes resident internal-state flow.
*/
func (op *MetalMarkovBlanket) FlowInternalTensor(
	sensory computetensor.Tensor,
	weights computetensor.Tensor,
	bias computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	sensoryTensor, weightTensor, biasTensor, err := markovThree(sensory, weights, bias)
	if err != nil {
		return nil, err
	}

	internalCount := outputShape.Len()
	sensoryCount := sensoryTensor.Len()
	if internalCount <= 0 || sensoryCount <= 0 ||
		weightTensor.Len() != internalCount*sensoryCount ||
		biasTensor.Len() != internalCount {
		return nil, fmt.Errorf("metal markov blanket: internal flow shape mismatch")
	}

	output, err := op.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_mb_flow_internal_tensor(
		sensoryTensor.buffer,
		weightTensor.buffer,
		biasTensor.buffer,
		output.buffer,
		C.int(internalCount),
		C.int(sensoryCount),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_mb_flow_internal_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
FlowActiveTensor computes resident active-state flow.
*/
func (op *MetalMarkovBlanket) FlowActiveTensor(
	internal computetensor.Tensor,
	weights computetensor.Tensor,
	bias computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	internalTensor, weightTensor, biasTensor, err := markovThree(internal, weights, bias)
	if err != nil {
		return nil, err
	}

	activeCount := outputShape.Len()
	internalCount := internalTensor.Len()
	if activeCount <= 0 || internalCount <= 0 ||
		weightTensor.Len() != activeCount*internalCount ||
		biasTensor.Len() != activeCount {
		return nil, fmt.Errorf("metal markov blanket: active flow shape mismatch")
	}

	output, err := op.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_mb_flow_active_tensor(
		internalTensor.buffer,
		weightTensor.buffer,
		biasTensor.buffer,
		output.buffer,
		C.int(activeCount),
		C.int(internalCount),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_mb_flow_active_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
MutualInformationTensor computes the resident Gaussian mutual information estimate.
*/
func (op *MetalMarkovBlanket) MutualInformationTensor(
	x computetensor.Tensor,
	y computetensor.Tensor,
	outputShape computetensor.Shape,
	xDimensions int,
	yDimensions int,
) (computetensor.Tensor, error) {
	xTensor, yTensor, err := markovTwo(x, y)
	if err != nil {
		return nil, err
	}

	if xDimensions <= 0 || yDimensions <= 0 || outputShape.Len() != 1 ||
		xTensor.Len()%xDimensions != 0 {
		return nil, fmt.Errorf("metal markov blanket: mutual information shape mismatch")
	}

	sampleCount := xTensor.Len() / xDimensions
	if sampleCount < 2 || yTensor.Len() != sampleCount*yDimensions {
		return nil, fmt.Errorf("metal markov blanket: mutual information sample mismatch")
	}

	output, err := op.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_mb_mutual_information_tensor(
		xTensor.buffer,
		yTensor.buffer,
		output.buffer,
		C.int(sampleCount),
		C.int(xDimensions),
		C.int(yDimensions),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_mb_mutual_information_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func markovTwo(
	first computetensor.Tensor,
	second computetensor.Tensor,
) (*Tensor, *Tensor, error) {
	firstTensor, err := requireMetalTensor(first)
	if err != nil {
		return nil, nil, err
	}

	secondTensor, err := requireMetalTensor(second)
	if err != nil {
		return nil, nil, err
	}

	return firstTensor, secondTensor, nil
}

func markovThree(
	first computetensor.Tensor,
	second computetensor.Tensor,
	third computetensor.Tensor,
) (*Tensor, *Tensor, *Tensor, error) {
	firstTensor, secondTensor, err := markovTwo(first, second)
	if err != nil {
		return nil, nil, nil, err
	}

	thirdTensor, err := requireMetalTensor(third)
	if err != nil {
		return nil, nil, nil, err
	}

	return firstTensor, secondTensor, thirdTensor, nil
}
