//go:build darwin && cgo

package metal

// #include "predictive_coding.h"
import "C"

import (
	"fmt"
	"unsafe"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (predictiveOps *MetalPredictiveCodingOps) PredictionTensor(
	weights computetensor.Tensor,
	representation computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	weightTensor, representationTensor, err := predictiveTwo(weights, representation)
	if err != nil {
		return nil, err
	}

	outFeatures := outputShape.Len()
	inFeatures := representationTensor.Len()
	if outFeatures <= 0 || inFeatures <= 0 || weightTensor.Len() != outFeatures*inFeatures {
		return nil, fmt.Errorf("metal predictive coding: prediction shape mismatch")
	}

	output, err := predictiveOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_pc_prediction_tensor(
		weightTensor.buffer,
		representationTensor.buffer,
		output.buffer,
		C.int(outFeatures),
		C.int(inFeatures),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_pc_prediction_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (predictiveOps *MetalPredictiveCodingOps) PredictionErrorTensor(
	observation computetensor.Tensor,
	prediction computetensor.Tensor,
	precision computetensor.Tensor,
) (computetensor.Tensor, error) {
	observationTensor, predictionTensor, err := predictiveTwo(observation, prediction)
	if err != nil {
		return nil, err
	}

	var precisionBuffer unsafe.Pointer
	if precision != nil {
		precisionTensor, precisionErr := requireMetalTensor(precision)
		if precisionErr != nil {
			return nil, precisionErr
		}

		if !observationTensor.shape.Equal(precisionTensor.shape) {
			return nil, fmt.Errorf("metal predictive coding: precision shape mismatch")
		}

		precisionBuffer = precisionTensor.buffer
	}

	output, err := predictiveOps.runtime.NewFloat32Tensor(observationTensor.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_pc_prediction_error_tensor(
		observationTensor.buffer,
		predictionTensor.buffer,
		precisionBuffer,
		output.buffer,
		C.int(observationTensor.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_pc_prediction_error_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (predictiveOps *MetalPredictiveCodingOps) UpdateRepresentationTensor(
	representation computetensor.Tensor,
	weights computetensor.Tensor,
	lowerError computetensor.Tensor,
	selfError computetensor.Tensor,
	learningRate computetensor.Tensor,
) (computetensor.Tensor, error) {
	representationTensor, weightTensor, lowerErrorTensor, selfErrorTensor, err := predictiveFour(
		representation,
		weights,
		lowerError,
		selfError,
	)
	if err != nil {
		return nil, err
	}

	learningRateTensor, err := requireMetalTensor(learningRate)
	if err != nil {
		return nil, err
	}

	inFeatures := representationTensor.Len()
	outFeatures := lowerErrorTensor.Len()
	if inFeatures <= 0 || outFeatures <= 0 ||
		weightTensor.Len() != outFeatures*inFeatures ||
		selfErrorTensor.Len() != inFeatures || learningRateTensor.Len() < 1 {
		return nil, fmt.Errorf("metal predictive coding: representation update shape mismatch")
	}

	output, err := predictiveOps.runtime.NewFloat32Tensor(representationTensor.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_pc_update_representation_tensor(
		representationTensor.buffer,
		weightTensor.buffer,
		lowerErrorTensor.buffer,
		selfErrorTensor.buffer,
		learningRateTensor.buffer,
		output.buffer,
		C.int(outFeatures),
		C.int(inFeatures),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_pc_update_representation_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (predictiveOps *MetalPredictiveCodingOps) UpdateWeightsTensor(
	weights computetensor.Tensor,
	predictionError computetensor.Tensor,
	representation computetensor.Tensor,
	outputShape computetensor.Shape,
	learningRate computetensor.Tensor,
) (computetensor.Tensor, error) {
	weightTensor, errorTensor, representationTensor, err := predictiveThree(
		weights,
		predictionError,
		representation,
	)
	if err != nil {
		return nil, err
	}

	learningRateTensor, err := requireMetalTensor(learningRate)
	if err != nil {
		return nil, err
	}

	outFeatures := errorTensor.Len()
	inFeatures := representationTensor.Len()
	if outFeatures <= 0 || inFeatures <= 0 ||
		weightTensor.Len() != outFeatures*inFeatures ||
		outputShape.Len() != weightTensor.Len() || learningRateTensor.Len() < 1 {
		return nil, fmt.Errorf("metal predictive coding: weight update shape mismatch")
	}

	output, err := predictiveOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_pc_update_weights_tensor(
		weightTensor.buffer,
		errorTensor.buffer,
		representationTensor.buffer,
		learningRateTensor.buffer,
		output.buffer,
		C.int(outFeatures),
		C.int(inFeatures),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_pc_update_weights_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func predictiveTwo(
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

func predictiveThree(
	first computetensor.Tensor,
	second computetensor.Tensor,
	third computetensor.Tensor,
) (*Tensor, *Tensor, *Tensor, error) {
	firstTensor, secondTensor, err := predictiveTwo(first, second)
	if err != nil {
		return nil, nil, nil, err
	}

	thirdTensor, err := requireMetalTensor(third)
	if err != nil {
		return nil, nil, nil, err
	}

	return firstTensor, secondTensor, thirdTensor, nil
}

func predictiveFour(
	first computetensor.Tensor,
	second computetensor.Tensor,
	third computetensor.Tensor,
	fourth computetensor.Tensor,
) (*Tensor, *Tensor, *Tensor, *Tensor, error) {
	firstTensor, secondTensor, thirdTensor, err := predictiveThree(first, second, third)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	fourthTensor, err := requireMetalTensor(fourth)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return firstTensor, secondTensor, thirdTensor, fourthTensor, nil
}
