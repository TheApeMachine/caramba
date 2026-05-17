//go:build darwin && cgo

package metal

// #include "metal_kernel_math.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (mathOps *MathOps) MSELossTensor(
	predictions computetensor.Float64Tensor,
	targets computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	predictionTensor, targetTensor, err := trainingPair(predictions, targets)
	if err != nil {
		return nil, err
	}

	outputShape, err := computetensor.NewShape([]int{1})
	if err != nil {
		return nil, err
	}

	output, err := mathOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_train_mse_loss_tensor(
		predictionTensor.buffer,
		targetTensor.buffer,
		output.buffer,
		C.int(predictionTensor.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_train_mse_loss_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (mathOps *MathOps) CrossEntropyLossTensor(
	logits computetensor.Float64Tensor,
	targets computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	logitTensor, targetTensor, err := trainingPair(logits, targets)
	if err != nil {
		return nil, err
	}

	if logitTensor.Len() == 0 {
		return nil, fmt.Errorf("metal tensor: cross entropy loss requires non-empty logits")
	}

	outputShape, err := computetensor.NewShape([]int{1})
	if err != nil {
		return nil, err
	}

	output, err := mathOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_train_cross_entropy_loss_tensor(
		logitTensor.buffer,
		targetTensor.buffer,
		output.buffer,
		C.int(logitTensor.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_train_cross_entropy_loss_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (mathOps *MathOps) MSEGradTensor(
	predictions computetensor.Float64Tensor,
	targets computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	predictionTensor, targetTensor, err := trainingPair(predictions, targets)
	if err != nil {
		return nil, err
	}

	output, err := mathOps.runtime.NewFloat32Tensor(predictionTensor.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_train_mse_grad_tensor(
		predictionTensor.buffer,
		targetTensor.buffer,
		output.buffer,
		C.int(predictionTensor.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_train_mse_grad_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (mathOps *MathOps) CrossEntropyGradTensor(
	logits computetensor.Float64Tensor,
	targets computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	logitTensor, targetTensor, err := trainingPair(logits, targets)
	if err != nil {
		return nil, err
	}

	output, err := mathOps.runtime.NewFloat32Tensor(logitTensor.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_train_cross_entropy_grad_tensor(
		logitTensor.buffer,
		targetTensor.buffer,
		output.buffer,
		C.int(logitTensor.Len()),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_train_cross_entropy_grad_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func trainingPair(
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

	if firstTensor.Len() != secondTensor.Len() {
		return nil, nil, fmt.Errorf("metal tensor: training input length mismatch")
	}

	return firstTensor, secondTensor, nil
}
