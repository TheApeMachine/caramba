//go:build darwin && cgo

package metal

// #include "hawkes.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
IntensityTensor computes resident Hawkes intensities.
*/
func (metalHawkes *MetalHawkes) IntensityTensor(
	times computetensor.Tensor,
	alpha computetensor.Tensor,
	beta computetensor.Tensor,
	mu computetensor.Tensor,
	currentTime computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	timesTensor, alphaTensor, betaTensor, muTensor, timeTensor, err := hawkesFive(
		times,
		alpha,
		beta,
		mu,
		currentTime,
	)
	if err != nil {
		return nil, err
	}

	processCount := alphaTensor.Len()
	eventCount := timesTensor.Len()

	if processCount <= 0 || outputShape.Len() != processCount ||
		betaTensor.Len() != processCount || muTensor.Len() != processCount || timeTensor.Len() < 1 {
		return nil, fmt.Errorf("metal hawkes: intensity tensor shape mismatch")
	}

	output, err := metalHawkes.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_hawkes_intensity_tensor(
		timesTensor.buffer,
		alphaTensor.buffer,
		betaTensor.buffer,
		muTensor.buffer,
		timeTensor.buffer,
		output.buffer,
		C.int(processCount),
		C.int(eventCount),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_hawkes_intensity_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
KernelMatrixTensor builds the resident Hawkes excitation matrix.
*/
func (metalHawkes *MetalHawkes) KernelMatrixTensor(
	times computetensor.Tensor,
	alpha computetensor.Tensor,
	beta computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	timesTensor, alphaTensor, betaTensor, err := hawkesThree(times, alpha, beta)
	if err != nil {
		return nil, err
	}

	eventCount := timesTensor.Len()
	if eventCount <= 0 || alphaTensor.Len() < 1 || betaTensor.Len() < 1 ||
		outputShape.Len() != eventCount*eventCount {
		return nil, fmt.Errorf("metal hawkes: kernel matrix tensor shape mismatch")
	}

	output, err := metalHawkes.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_hawkes_kernel_matrix_tensor(
		timesTensor.buffer,
		alphaTensor.buffer,
		betaTensor.buffer,
		output.buffer,
		C.int(eventCount),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_hawkes_kernel_matrix_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
LogLikelihoodTensor computes resident Hawkes log-likelihood.
*/
func (metalHawkes *MetalHawkes) LogLikelihoodTensor(
	intensities computetensor.Tensor,
	integral computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	intensityTensor, integralTensor, err := hawkesTwo(intensities, integral)
	if err != nil {
		return nil, err
	}

	eventCount := intensityTensor.Len()
	if eventCount <= 0 || integralTensor.Len() < 1 || outputShape.Len() != 1 {
		return nil, fmt.Errorf("metal hawkes: log likelihood tensor shape mismatch")
	}

	output, err := metalHawkes.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_hawkes_log_likelihood_tensor(
		intensityTensor.buffer,
		integralTensor.buffer,
		output.buffer,
		C.int(eventCount),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_hawkes_log_likelihood_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

/*
SimulateTensor runs resident Hawkes simulation into Metal storage.
*/
func (metalHawkes *MetalHawkes) SimulateTensor(
	mu computetensor.Tensor,
	alpha computetensor.Tensor,
	beta computetensor.Tensor,
	tMax computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	muTensor, alphaTensor, betaTensor, tMaxTensor, err := hawkesFour(mu, alpha, beta, tMax)
	if err != nil {
		return nil, err
	}

	processCount := muTensor.Len()
	if processCount <= 0 || alphaTensor.Len() != processCount || betaTensor.Len() != processCount ||
		tMaxTensor.Len() < 1 || outputShape.Len()%processCount != 0 {
		return nil, fmt.Errorf("metal hawkes: simulate tensor shape mismatch")
	}

	maxSteps := outputShape.Len() / processCount
	output, err := metalHawkes.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_hawkes_simulate_tensor(
		muTensor.buffer,
		alphaTensor.buffer,
		betaTensor.buffer,
		tMaxTensor.buffer,
		C.int(processCount),
		C.int(maxSteps),
		output.buffer,
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_hawkes_simulate_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func hawkesTwo(
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

func hawkesThree(
	first computetensor.Tensor,
	second computetensor.Tensor,
	third computetensor.Tensor,
) (*Tensor, *Tensor, *Tensor, error) {
	firstTensor, secondTensor, err := hawkesTwo(first, second)
	if err != nil {
		return nil, nil, nil, err
	}

	thirdTensor, err := requireMetalTensor(third)
	if err != nil {
		return nil, nil, nil, err
	}

	return firstTensor, secondTensor, thirdTensor, nil
}

func hawkesFour(
	first computetensor.Tensor,
	second computetensor.Tensor,
	third computetensor.Tensor,
	fourth computetensor.Tensor,
) (*Tensor, *Tensor, *Tensor, *Tensor, error) {
	firstTensor, secondTensor, thirdTensor, err := hawkesThree(first, second, third)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	fourthTensor, err := requireMetalTensor(fourth)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	return firstTensor, secondTensor, thirdTensor, fourthTensor, nil
}

func hawkesFive(
	first computetensor.Tensor,
	second computetensor.Tensor,
	third computetensor.Tensor,
	fourth computetensor.Tensor,
	fifth computetensor.Tensor,
) (*Tensor, *Tensor, *Tensor, *Tensor, *Tensor, error) {
	firstTensor, secondTensor, thirdTensor, fourthTensor, err := hawkesFour(
		first,
		second,
		third,
		fourth,
	)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	fifthTensor, err := requireMetalTensor(fifth)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}

	return firstTensor, secondTensor, thirdTensor, fourthTensor, fifthTensor, nil
}
