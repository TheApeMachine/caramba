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

const metalCausalThreadCountGo = 256

func runMetalBackdoorAdjustment(
	conditional tensor.Tensor,
	marginal tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalBackdoorAdjustment(conditional, marginal, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.first, config.second)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_backdoor_adjustment(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.out.buffer, C.uint32_t(config.rows),
		C.uint32_t(config.inner), C.uint32_t(config.cols), C.uint64_t(token), &status,
	)

	return finishMetalCausalDispatch("backdoor_adjustment", token, rc, status)
}

func runMetalFrontdoorAdjustment(
	mediator tensor.Tensor,
	outcome tensor.Tensor,
	marginal tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalFrontdoorAdjustment(mediator, outcome, marginal, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.first, config.second, config.third)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_frontdoor_adjustment(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.third.buffer, config.out.buffer, C.uint32_t(config.rows),
		C.uint32_t(config.inner), C.uint32_t(config.cols), C.uint64_t(token), &status,
	)

	return finishMetalCausalDispatch("frontdoor_adjustment", token, rc, status)
}

func runMetalDoIntervene(adjacency tensor.Tensor, intervened tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalDoIntervene(adjacency, intervened, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.first, config.second)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_do_intervene(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.out.buffer, C.uint32_t(config.rows),
		C.uint32_t(config.inner), C.uint64_t(token), &status,
	)

	return finishMetalCausalDispatch("do_intervene", token, rc, status)
}

func runMetalCATE(treated tensor.Tensor, control tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalCATE(treated, control, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.first, config.second)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_cate(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.out.buffer, C.uint32_t(config.count), C.uint64_t(token), &status,
	)

	return finishMetalCausalDispatch("cate", token, rc, status)
}

func runMetalCounterfactual(
	observedY tensor.Tensor,
	observedX tensor.Tensor,
	counterfactualX tensor.Tensor,
	slope tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalCounterfactual(observedY, observedX, counterfactualX, slope, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.first, config.second, config.third, config.fourth)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_counterfactual(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.third.buffer, config.fourth.buffer, config.out.buffer,
		C.uint32_t(config.count), C.uint64_t(token), &status,
	)

	return finishMetalCausalDispatch("counterfactual", token, rc, status)
}

func runMetalIVEstimate(
	instrument tensor.Tensor,
	treatment tensor.Tensor,
	outcome tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalIVEstimate(instrument, treatment, outcome, out)
	if err != nil {
		return err
	}

	if err := config.allocateScratch(5); err != nil {
		return err
	}

	token, err := config.beginCompletion(config.first, config.second, config.third)
	if err != nil {
		config.closeScratch()
		return err
	}

	config.closeScratch()
	status := C.MetalStatus{}
	rc := C.metal_dispatch_iv_estimate(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.third.buffer, config.scratch.buffer, config.out.buffer,
		C.uint32_t(config.count), C.uint32_t(config.partialCount), C.uint64_t(token), &status,
	)

	return finishMetalCausalDispatch("iv_estimate", token, rc, status)
}

func runMetalDAGMarkovFactorization(
	conditionals tensor.Tensor,
	parents tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalDAGMarkovFactorization(conditionals, parents, out)
	if err != nil {
		return err
	}

	if err := config.allocateScratch(1); err != nil {
		return err
	}

	token, err := config.beginCompletion(config.first, config.second)
	if err != nil {
		config.closeScratch()
		return err
	}

	config.closeScratch()
	status := C.MetalStatus{}
	rc := C.metal_dispatch_dag_markov_factorization(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.scratch.buffer, config.out.buffer, C.uint32_t(config.count),
		C.uint32_t(config.partialCount), C.uint64_t(token), &status,
	)

	return finishMetalCausalDispatch("dag_markov_factorization", token, rc, status)
}

func metalCausalPartialCount(elementCount int) int {
	return (elementCount + metalCausalThreadCountGo - 1) / metalCausalThreadCountGo
}

func newMetalCausalScratch(bridge *metalBridge, partialCount int, valuesPerPartial int) (*metalTensor, error) {
	shape, err := tensor.NewShape([]int{partialCount * valuesPerPartial})
	if err != nil {
		return nil, err
	}

	return bridge.empty(shape, dtype.Float32)
}

func (config *metalCausalScalarConfig) allocateScratch(valuesPerPartial int) error {
	scratch, err := newMetalCausalScratch(config.out.bridge, int(config.partialCount), valuesPerPartial)
	if err != nil {
		return err
	}

	config.scratch = scratch
	return nil
}

func (config metalCausalScalarConfig) beginCompletion(sources ...*metalTensor) (uint64, error) {
	return metalCompletions.BeginMany([]*metalTensor{config.out, config.scratch}, sources...)
}

func (config metalCausalScalarConfig) closeScratch() {
	_ = config.scratch.Close()
}

func finishMetalCausalDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
