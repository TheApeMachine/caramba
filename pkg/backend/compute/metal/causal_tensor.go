//go:build darwin && cgo

package metal

// #include "causal.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
CounterfactualTensor evaluates resident linear SCM counterfactuals.
*/
func (metalCausalOps *MetalCausalOps) CounterfactualTensor(
	observedX computetensor.Float64Tensor,
	observedY computetensor.Float64Tensor,
	beta computetensor.Float64Tensor,
	counterfactualX computetensor.Float64Tensor,
	outputShape computetensor.Shape,
) (computetensor.Float64Tensor, error) {
	xTensor, yTensor, betaTensor, cfTensor, err := causalFour(
		observedX,
		observedY,
		beta,
		counterfactualX,
	)
	if err != nil {
		return nil, err
	}

	observedCount := xTensor.Len()
	counterfactualCount := cfTensor.Len()
	if observedCount <= 0 || counterfactualCount <= 0 ||
		yTensor.Len() != observedCount || betaTensor.Len() != observedCount ||
		outputShape.Len() != observedCount*counterfactualCount {
		return nil, fmt.Errorf("metal causal: counterfactual tensor shape mismatch")
	}

	output, err := metalCausalOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_causal_counterfactual_tensor(
		xTensor.buffer,
		yTensor.buffer,
		betaTensor.buffer,
		cfTensor.buffer,
		output.buffer,
		C.int(observedCount),
		C.int(counterfactualCount),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_causal_counterfactual_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
FrontdoorAdjustmentTensor computes resident frontdoor causal effects.
*/
func (metalCausalOps *MetalCausalOps) FrontdoorAdjustmentTensor(
	treatment computetensor.Float64Tensor,
	mediator computetensor.Float64Tensor,
	outcome computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	samples int,
	treatmentBins int,
	mediatorBins int,
) (computetensor.Float64Tensor, error) {
	treatmentTensor, mediatorTensor, outcomeTensor, err := causalThree(treatment, mediator, outcome)
	if err != nil {
		return nil, err
	}

	if samples <= 0 || treatmentBins <= 0 || mediatorBins <= 0 ||
		treatmentTensor.Len() < samples || mediatorTensor.Len() < samples ||
		outcomeTensor.Len() < samples || outputShape.Len() != treatmentBins {
		return nil, fmt.Errorf("metal causal: frontdoor tensor shape mismatch")
	}

	output, err := metalCausalOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_causal_frontdoor_tensor(
		treatmentTensor.buffer,
		mediatorTensor.buffer,
		outcomeTensor.buffer,
		output.buffer,
		C.int(samples),
		C.int(treatmentBins),
		C.int(mediatorBins),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_causal_frontdoor_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func causalTwo(
	first computetensor.Float64Tensor,
	second computetensor.Float64Tensor,
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

func causalThree(
	first computetensor.Float64Tensor,
	second computetensor.Float64Tensor,
	third computetensor.Float64Tensor,
) (*Tensor, *Tensor, *Tensor, error) {
	firstTensor, secondTensor, err := causalTwo(first, second)
	if err != nil {
		return nil, nil, nil, err
	}

	thirdTensor, err := requireMetalTensor(third)
	if err != nil {
		return nil, nil, nil, err
	}

	return firstTensor, secondTensor, thirdTensor, nil
}

func causalFour(
	first computetensor.Float64Tensor,
	second computetensor.Float64Tensor,
	third computetensor.Float64Tensor,
	fourth computetensor.Float64Tensor,
) (*Tensor, *Tensor, *Tensor, *Tensor, error) {
	firstTensor, secondTensor, thirdTensor, err := causalThree(first, second, third)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	fourthTensor, err := requireMetalTensor(fourth)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return firstTensor, secondTensor, thirdTensor, fourthTensor, nil
}
