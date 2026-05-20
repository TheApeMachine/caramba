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

func runMetalResearchUnaryKernel(
	operation metalResearchOp,
	input tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalResearchUnary(input, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_research_unary(
		config.input.bridge.device,
		C.int(operation),
		C.int(config.elementDType),
		config.input.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalResearchDispatch("research unary", token, rc, status)
}

func runMetalResearchBinaryKernel(
	operation metalResearchOp,
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalResearchBinary(left, right, out)
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
	rc := C.metal_dispatch_research_binary(
		config.left.bridge.device,
		C.int(operation),
		C.int(config.elementDType),
		config.left.buffer,
		config.right.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalResearchDispatch("research binary", token, rc, status)
}

func runMetalPCPrediction(
	weights tensor.Tensor,
	state tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalPCPrediction(weights, state, out)
	if err != nil {
		return err
	}

	if config.outCount == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.weights, config.state)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_pc_prediction(
		config.weights.bridge.device,
		C.int(config.elementDType),
		config.weights.buffer,
		config.state.buffer,
		config.out.buffer,
		C.uint32_t(config.outCount),
		C.uint32_t(config.inCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalResearchDispatch("pc_prediction", token, rc, status)
}

func runMetalPCUpdateRepresentation(
	weights tensor.Tensor,
	state tensor.Tensor,
	predictionError tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalPCUpdate(weights, state, predictionError, out, false)
	if err != nil {
		return err
	}

	if config.inCount == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.weights, config.state, config.error)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_pc_update_representation(
		config.weights.bridge.device,
		C.int(config.elementDType),
		config.weights.buffer,
		config.state.buffer,
		config.error.buffer,
		config.out.buffer,
		C.uint32_t(config.outCount),
		C.uint32_t(config.inCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalResearchDispatch("pc_update_representation", token, rc, status)
}

func runMetalPCUpdateWeights(
	weights tensor.Tensor,
	state tensor.Tensor,
	predictionError tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalPCUpdate(weights, state, predictionError, out, true)
	if err != nil {
		return err
	}

	if config.outCount == 0 || config.inCount == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.weights, config.state, config.error)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_pc_update_weights(
		config.weights.bridge.device,
		C.int(config.elementDType),
		config.weights.buffer,
		config.state.buffer,
		config.error.buffer,
		config.out.buffer,
		C.uint32_t(config.outCount),
		C.uint32_t(config.inCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalResearchDispatch("pc_update_weights", token, rc, status)
}

func finishMetalResearchDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
