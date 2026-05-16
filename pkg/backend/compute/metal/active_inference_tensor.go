//go:build darwin && cgo

package metal

// #include "active_inference.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
FreeEnergyTensor computes resident variational free energy.
*/
func (activeInferenceOps *ActiveInferenceOps) FreeEnergyTensor(
	mu computetensor.Float64Tensor,
	logSigma computetensor.Float64Tensor,
	outputShape computetensor.Shape,
) (computetensor.Float64Tensor, error) {
	muTensor, logSigmaTensor, err := requireActiveInferenceBinary(mu, logSigma)
	if err != nil {
		return nil, err
	}

	if outputShape.Len() != 1 {
		return nil, fmt.Errorf("metal active inference: free energy output length must be 1")
	}

	output, err := activeInferenceOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_ai_free_energy_tensor(
		muTensor.buffer,
		logSigmaTensor.buffer,
		output.buffer,
		C.int(muTensor.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_ai_free_energy_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
BeliefUpdateTensor computes one resident active-inference belief update.
*/
func (activeInferenceOps *ActiveInferenceOps) BeliefUpdateTensor(
	mu computetensor.Float64Tensor,
	logSigma computetensor.Float64Tensor,
	predictionError computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	learningRate float32,
) (computetensor.Float64Tensor, error) {
	muTensor, logSigmaTensor, predictionErrorTensor, err := activeInferenceTriple(
		mu,
		logSigma,
		predictionError,
	)
	if err != nil {
		return nil, err
	}

	if outputShape.Len() != 2*muTensor.Len() {
		return nil, fmt.Errorf("metal active inference: belief update output shape mismatch")
	}

	output, err := activeInferenceOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_ai_belief_update_tensor(
		muTensor.buffer,
		logSigmaTensor.buffer,
		predictionErrorTensor.buffer,
		C.float(learningRate),
		output.buffer,
		C.int(muTensor.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_ai_belief_update_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
PrecisionWeightTensor computes resident precision-weighted error.
*/
func (activeInferenceOps *ActiveInferenceOps) PrecisionWeightTensor(
	errTensor computetensor.Float64Tensor,
	logPrecision computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	metalError, metalLogPrecision, err := requireActiveInferenceBinary(errTensor, logPrecision)
	if err != nil {
		return nil, err
	}

	output, err := activeInferenceOps.runtime.NewFloat32Tensor(metalError.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_ai_precision_weight_tensor(
		metalError.buffer,
		metalLogPrecision.buffer,
		output.buffer,
		C.int(metalError.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_ai_precision_weight_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
ExpectedFreeEnergyTensor computes resident expected free energy by policy column.
*/
func (activeInferenceOps *ActiveInferenceOps) ExpectedFreeEnergyTensor(
	qOutcomes computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	outcomeCount int,
	policyCount int,
) (computetensor.Float64Tensor, error) {
	qTensor, err := requireMetalTensor(qOutcomes)
	if err != nil {
		return nil, err
	}

	if outcomeCount <= 0 || policyCount <= 0 ||
		qTensor.Len() != outcomeCount*policyCount || outputShape.Len() != policyCount {
		return nil, fmt.Errorf("metal active inference: expected free energy shape mismatch")
	}

	output, err := activeInferenceOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_ai_expected_free_energy_tensor(
		qTensor.buffer,
		output.buffer,
		C.int(outcomeCount),
		C.int(policyCount),
		C.float(DefaultExpectedFreeEnergyEps),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_ai_expected_free_energy_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func requireActiveInferenceBinary(
	left computetensor.Float64Tensor,
	right computetensor.Float64Tensor,
) (*Tensor, *Tensor, error) {
	metalLeft, err := requireMetalTensor(left)
	if err != nil {
		return nil, nil, err
	}

	metalRight, err := requireMetalTensor(right)
	if err != nil {
		return nil, nil, err
	}

	if !metalLeft.shape.Equal(metalRight.shape) {
		return nil, nil, fmt.Errorf("metal active inference: input shape mismatch")
	}

	return metalLeft, metalRight, nil
}

func activeInferenceTriple(
	first computetensor.Float64Tensor,
	second computetensor.Float64Tensor,
	third computetensor.Float64Tensor,
) (*Tensor, *Tensor, *Tensor, error) {
	firstTensor, secondTensor, err := requireActiveInferenceBinary(first, second)
	if err != nil {
		return nil, nil, nil, err
	}

	thirdTensor, err := requireMetalTensor(third)
	if err != nil {
		return nil, nil, nil, err
	}

	if !firstTensor.shape.Equal(thirdTensor.shape) {
		return nil, nil, nil, fmt.Errorf("metal active inference: input shape mismatch")
	}

	return firstTensor, secondTensor, thirdTensor, nil
}
