//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

const metalActiveThreadCountGo = 256

func runMetalFreeEnergy(
	likelihood tensor.Tensor,
	posterior tensor.Tensor,
	prior tensor.Tensor,
	auxiliary tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalActiveFreeEnergy(likelihood, posterior, prior, auxiliary, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	if err := config.allocateScratch(); err != nil {
		return err
	}

	token, err := config.beginCompletion(config.likelihood, config.posterior, config.prior, config.auxiliary)
	if err != nil {
		config.closeScratch()
		return err
	}

	config.closeScratch()

	status := C.MetalStatus{}
	rc := C.metal_dispatch_active_free_energy(
		config.likelihood.bridge.device,
		C.int(config.elementDType),
		config.likelihood.buffer,
		config.posterior.buffer,
		config.prior.buffer,
		config.scratch.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint32_t(config.partialCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalActiveDispatch("free_energy", token, rc, status)
}

func runMetalExpectedFreeEnergy(
	predictedObs tensor.Tensor,
	preferredObs tensor.Tensor,
	predictedState tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalExpectedFreeEnergy(predictedObs, preferredObs, predictedState, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	if err := config.allocateScratch(); err != nil {
		return err
	}

	token, err := config.beginCompletion(config.first, config.second, config.third)
	if err != nil {
		config.closeScratch()
		return err
	}

	config.closeScratch()

	status := C.MetalStatus{}
	rc := C.metal_dispatch_expected_free_energy(
		config.first.bridge.device,
		C.int(config.elementDType),
		config.first.buffer,
		config.second.buffer,
		config.third.buffer,
		config.scratch.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint32_t(config.stateCount),
		C.uint32_t(config.partialCount),
		C.uint32_t(config.statePartialCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalActiveDispatch("expected_free_energy", token, rc, status)
}

func runMetalBeliefUpdate(
	likelihood tensor.Tensor,
	prior tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalActiveBinary(likelihood, prior, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	if err := config.allocateScratch(); err != nil {
		return err
	}

	token, err := config.beginCompletion(config.left, config.right)
	if err != nil {
		config.closeScratch()
		return err
	}

	config.closeScratch()

	status := C.MetalStatus{}
	rc := C.metal_dispatch_belief_update(
		config.left.bridge.device,
		C.int(config.elementDType),
		config.left.buffer,
		config.right.buffer,
		config.scratch.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint32_t(config.partialCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalActiveDispatch("belief_update", token, rc, status)
}

func runMetalPrecisionWeight(
	errors tensor.Tensor,
	precision tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalActiveBinary(errors, precision, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.left, config.right)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_precision_weight(
		config.left.bridge.device,
		C.int(config.elementDType),
		config.left.buffer,
		config.right.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalActiveDispatch("precision_weight", token, rc, status)
}

func metalActivePartialCount(elementCount int) int {
	return (elementCount + metalActiveThreadCountGo - 1) / metalActiveThreadCountGo
}

func newMetalActiveScratch(bridge *metalBridge, partialCount int) (*metalTensor, error) {
	shape, err := tensor.NewShape([]int{partialCount})
	if err != nil {
		return nil, err
	}

	return bridge.empty(shape, dtype.Float32)
}

func finishMetalActiveDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
