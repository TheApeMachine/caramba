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

	"github.com/theapemachine/manifesto/tensor"
)

type metalPhysicsBinaryConfig struct {
	input        *metalTensor
	spacing      *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
	rank         uint32
	dim0         uint32
	dim1         uint32
	dim2         uint32
}

type metalPhysicsTernaryConfig struct {
	first        *metalTensor
	second       *metalTensor
	third        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
}

type metalPhysicsFFTConfig struct {
	realIn       *metalTensor
	imagIn       *metalTensor
	realOut      *metalTensor
	imagOut      *metalTensor
	elementDType metalElementDType
	count        uint32
}

func runMetalPhysicsBinary(
	operation metalPhysicsBinaryOp,
	input tensor.Tensor,
	spacing tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalPhysicsBinary(operation, input, spacing, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.spacing)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := dispatchMetalPhysicsBinary(operation, config, token, &status)
	return finishMetalPhysicsDispatch(operation.String(), token, rc, status)
}

func runMetalFFT1D(
	realIn tensor.Tensor,
	imagIn tensor.Tensor,
	realOut tensor.Tensor,
	imagOut tensor.Tensor,
) error {
	return runMetalFFT(realIn, imagIn, realOut, imagOut, false)
}

func runMetalIFFT1D(
	realIn tensor.Tensor,
	imagIn tensor.Tensor,
	realOut tensor.Tensor,
	imagOut tensor.Tensor,
) error {
	return runMetalFFT(realIn, imagIn, realOut, imagOut, true)
}

func runMetalFFT(
	realIn tensor.Tensor,
	imagIn tensor.Tensor,
	realOut tensor.Tensor,
	imagOut tensor.Tensor,
	inverse bool,
) error {
	config, err := requireMetalPhysicsFFT(realIn, imagIn, realOut, imagOut)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.BeginMany(
		[]*metalTensor{config.realOut, config.imagOut},
		config.realIn, config.imagIn,
	)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_fft1d(
		config.realIn.bridge.device, C.int(config.elementDType), config.realIn.buffer,
		config.imagIn.buffer, config.realOut.buffer, config.imagOut.buffer,
		C.uint32_t(config.count), C.bool(inverse), C.uint64_t(token), &status,
	)

	if inverse {
		return finishMetalPhysicsDispatch("ifft1d", token, rc, status)
	}

	return finishMetalPhysicsDispatch("fft1d", token, rc, status)
}

func runMetalMadelungContinuity(
	density tensor.Tensor,
	velocity tensor.Tensor,
	spacing tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalMadelungContinuity(density, velocity, spacing, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.first, config.second, config.third)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_madelung_continuity(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.third.buffer, config.out.buffer,
		C.uint32_t(config.count), C.uint64_t(token), &status,
	)

	return finishMetalPhysicsDispatch("madelung_continuity", token, rc, status)
}

func dispatchMetalPhysicsBinary(
	operation metalPhysicsBinaryOp,
	config metalPhysicsBinaryConfig,
	token uint64,
	status *C.MetalStatus,
) C.int {
	switch operation {
	case metalPhysicsLaplacian:
		return C.metal_dispatch_laplacian(
			config.input.bridge.device, C.int(config.elementDType), config.input.buffer,
			config.spacing.buffer, config.out.buffer, C.uint32_t(config.count),
			C.uint32_t(config.rank), C.uint32_t(config.dim0), C.uint32_t(config.dim1),
			C.uint32_t(config.dim2), C.uint64_t(token), status,
		)
	case metalPhysicsLaplacian4:
		return dispatchMetalPhysicsVector("laplacian4", config, token, status)
	case metalPhysicsGrad1D:
		return dispatchMetalPhysicsVector("grad1d", config, token, status)
	case metalPhysicsDivergence1D:
		return dispatchMetalPhysicsVector("divergence1d", config, token, status)
	case metalPhysicsQuantumPotential:
		return dispatchMetalPhysicsVector("quantum_potential", config, token, status)
	default:
		return dispatchMetalPhysicsVector("bohmian_velocity", config, token, status)
	}
}

func dispatchMetalPhysicsVector(
	name string,
	config metalPhysicsBinaryConfig,
	token uint64,
	status *C.MetalStatus,
) C.int {
	switch name {
	case "laplacian4":
		return C.metal_dispatch_laplacian4(
			config.input.bridge.device, C.int(config.elementDType), config.input.buffer,
			config.spacing.buffer, config.out.buffer, C.uint32_t(config.count),
			C.uint64_t(token), status,
		)
	case "grad1d":
		return C.metal_dispatch_grad1d(
			config.input.bridge.device, C.int(config.elementDType), config.input.buffer,
			config.spacing.buffer, config.out.buffer, C.uint32_t(config.count),
			C.uint64_t(token), status,
		)
	case "divergence1d":
		return C.metal_dispatch_divergence1d(
			config.input.bridge.device, C.int(config.elementDType), config.input.buffer,
			config.spacing.buffer, config.out.buffer, C.uint32_t(config.count),
			C.uint64_t(token), status,
		)
	case "quantum_potential":
		return C.metal_dispatch_quantum_potential(
			config.input.bridge.device, C.int(config.elementDType), config.input.buffer,
			config.spacing.buffer, config.out.buffer, C.uint32_t(config.count),
			C.uint64_t(token), status,
		)
	default:
		return C.metal_dispatch_bohmian_velocity(
			config.input.bridge.device, C.int(config.elementDType), config.input.buffer,
			config.spacing.buffer, config.out.buffer, C.uint32_t(config.count),
			C.uint64_t(token), status,
		)
	}
}

func finishMetalPhysicsDispatch(
	name string,
	token uint64,
	rc C.int,
	status C.MetalStatus,
) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}

func (operation metalPhysicsBinaryOp) String() string {
	switch operation {
	case metalPhysicsLaplacian:
		return "laplacian"
	case metalPhysicsLaplacian4:
		return "laplacian4"
	case metalPhysicsGrad1D:
		return "grad1d"
	case metalPhysicsDivergence1D:
		return "divergence1d"
	case metalPhysicsQuantumPotential:
		return "quantum_potential"
	default:
		return "bohmian_velocity"
	}
}
