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

/*
BackdoorAdjustmentTensor computes resident backdoor-adjusted causal effects.
*/
func (metalCausalOps *MetalCausalOps) BackdoorAdjustmentTensor(
	outcome computetensor.Float64Tensor,
	treatment computetensor.Float64Tensor,
	confounder computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	samples int,
	outcomeDimensions int,
	treatmentDimensions int,
	confounderDimensions int,
) (computetensor.Float64Tensor, error) {
	outcomeTensor, treatmentTensor, confounderTensor, err := causalThree(outcome, treatment, confounder)
	if err != nil {
		return nil, err
	}

	if samples <= 0 || outcomeDimensions <= 0 || treatmentDimensions <= 0 || confounderDimensions < 0 ||
		outcomeTensor.Len() < samples*outcomeDimensions ||
		treatmentTensor.Len() < samples*treatmentDimensions ||
		confounderTensor.Len() < samples*confounderDimensions ||
		outputShape.Len() != outcomeDimensions {
		return nil, fmt.Errorf("metal causal: backdoor tensor shape mismatch")
	}

	output, err := metalCausalOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_causal_backdoor_tensor(
		outcomeTensor.buffer,
		treatmentTensor.buffer,
		confounderTensor.buffer,
		output.buffer,
		C.int(samples),
		C.int(outcomeDimensions),
		C.int(treatmentDimensions),
		C.int(confounderDimensions),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_causal_backdoor_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
CATETensor computes resident conditional average treatment effects.
*/
func (metalCausalOps *MetalCausalOps) CATETensor(
	covariates computetensor.Float64Tensor,
	treatment computetensor.Float64Tensor,
	outcome computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	samples int,
	covariateDimensions int,
) (computetensor.Float64Tensor, error) {
	covariateTensor, treatmentTensor, outcomeTensor, err := causalThree(covariates, treatment, outcome)
	if err != nil {
		return nil, err
	}

	if samples <= 0 || covariateDimensions <= 0 ||
		covariateTensor.Len() < samples*covariateDimensions ||
		treatmentTensor.Len() < samples || outcomeTensor.Len() < samples ||
		outputShape.Len() != samples {
		return nil, fmt.Errorf("metal causal: cate tensor shape mismatch")
	}

	output, err := metalCausalOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_causal_cate_tensor(
		covariateTensor.buffer,
		treatmentTensor.buffer,
		outcomeTensor.buffer,
		output.buffer,
		C.int(samples),
		C.int(covariateDimensions),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_causal_cate_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
IVEstimateTensor computes resident two-stage least-squares estimates.
*/
func (metalCausalOps *MetalCausalOps) IVEstimateTensor(
	instrument computetensor.Float64Tensor,
	treatment computetensor.Float64Tensor,
	outcome computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	samples int,
	instrumentDimensions int,
	treatmentDimensions int,
	outcomeDimensions int,
) (computetensor.Float64Tensor, error) {
	instrumentTensor, treatmentTensor, outcomeTensor, err := causalThree(instrument, treatment, outcome)
	if err != nil {
		return nil, err
	}

	if samples <= 0 || instrumentDimensions <= 0 || treatmentDimensions <= 0 || outcomeDimensions <= 0 ||
		instrumentTensor.Len() < samples*instrumentDimensions ||
		treatmentTensor.Len() < samples*treatmentDimensions ||
		outcomeTensor.Len() < samples*outcomeDimensions ||
		outputShape.Len() != treatmentDimensions*outcomeDimensions {
		return nil, fmt.Errorf("metal causal: iv_estimate tensor shape mismatch")
	}

	output, err := metalCausalOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_causal_iv_tensor(
		instrumentTensor.buffer,
		treatmentTensor.buffer,
		outcomeTensor.buffer,
		output.buffer,
		C.int(samples),
		C.int(instrumentDimensions),
		C.int(treatmentDimensions),
		C.int(outcomeDimensions),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_causal_iv_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
DAGMarkovFactorizationTensor computes resident DAG Markov log probabilities.
*/
func (metalCausalOps *MetalCausalOps) DAGMarkovFactorizationTensor(
	observations computetensor.Float64Tensor,
	adjacency computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	nodeCount int,
	samples int,
) (computetensor.Float64Tensor, error) {
	observationTensor, adjacencyTensor, err := causalTwo(observations, adjacency)
	if err != nil {
		return nil, err
	}

	if nodeCount <= 0 || samples <= 0 ||
		observationTensor.Len() < samples*nodeCount ||
		adjacencyTensor.Len() < nodeCount*nodeCount ||
		outputShape.Len() != samples {
		return nil, fmt.Errorf("metal causal: dag_markov_factorization tensor shape mismatch")
	}

	output, err := metalCausalOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_causal_dag_markov_tensor(
		observationTensor.buffer,
		adjacencyTensor.buffer,
		output.buffer,
		C.int(samples),
		C.int(nodeCount),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_causal_dag_markov_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
DoCalculusTensor computes resident Gaussian do-calculus graph surgery.
*/
func (metalCausalOps *MetalCausalOps) DoCalculusTensor(
	covariance computetensor.Float64Tensor,
	interventionMask computetensor.Float64Tensor,
	interventionValues computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	nodeCount int,
) (computetensor.Float64Tensor, error) {
	covarianceTensor, maskTensor, valueTensor, err := causalThree(
		covariance,
		interventionMask,
		interventionValues,
	)
	if err != nil {
		return nil, err
	}

	if nodeCount <= 0 ||
		covarianceTensor.Len() < nodeCount*nodeCount ||
		maskTensor.Len() < nodeCount ||
		valueTensor.Len() < nodeCount ||
		outputShape.Len() != nodeCount+nodeCount*nodeCount {
		return nil, fmt.Errorf("metal causal: do_calculus tensor shape mismatch")
	}

	output, err := metalCausalOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_causal_do_calculus_tensor(
		covarianceTensor.buffer,
		maskTensor.buffer,
		valueTensor.buffer,
		output.buffer,
		C.int(nodeCount),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_causal_do_calculus_tensor failed (rc=%d)", rc)
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
