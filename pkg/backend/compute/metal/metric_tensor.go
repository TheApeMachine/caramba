//go:build darwin && cgo

package metal

// #include "metal_kernel_math.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (mathOps *MathOps) AccuracyTensor(
	predictions computetensor.Tensor,
	targets computetensor.Tensor,
) (computetensor.Tensor, error) {
	predictionTensor, targetTensor, err := trainingPair(predictions, targets)
	if err != nil {
		return nil, err
	}

	if predictionTensor.Len() == 0 {
		return nil, fmt.Errorf("metal tensor: accuracy requires non-empty inputs")
	}

	output, err := mathOps.newScalarTensor()
	if err != nil {
		return nil, err
	}

	rc := C.metal_bench_accuracy_tensor(
		predictionTensor.buffer,
		targetTensor.buffer,
		output.buffer,
		C.int(predictionTensor.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_bench_accuracy_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (mathOps *MathOps) PerplexityTensor(
	probabilities computetensor.Tensor,
	targets computetensor.Tensor,
) (computetensor.Tensor, error) {
	probabilityTensor, targetTensor, err := trainingPair(probabilities, targets)
	if err != nil {
		return nil, err
	}

	if probabilityTensor.Len() == 0 {
		return nil, fmt.Errorf("metal tensor: perplexity requires non-empty inputs")
	}

	output, err := mathOps.newScalarTensor()
	if err != nil {
		return nil, err
	}

	rc := C.metal_bench_perplexity_tensor(
		probabilityTensor.buffer,
		targetTensor.buffer,
		output.buffer,
		C.int(probabilityTensor.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_bench_perplexity_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (mathOps *MathOps) F1Tensor(
	predictions computetensor.Tensor,
	targets computetensor.Tensor,
) (computetensor.Tensor, error) {
	predictionTensor, targetTensor, err := trainingPair(predictions, targets)
	if err != nil {
		return nil, err
	}

	output, err := mathOps.newScalarTensor()
	if err != nil {
		return nil, err
	}

	rc := C.metal_bench_f1_tensor(
		predictionTensor.buffer,
		targetTensor.buffer,
		output.buffer,
		C.int(predictionTensor.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_bench_f1_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (mathOps *MathOps) newScalarTensor() (*Tensor, error) {
	outputShape, err := computetensor.NewShape([]int{1})
	if err != nil {
		return nil, err
	}

	return mathOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
}
