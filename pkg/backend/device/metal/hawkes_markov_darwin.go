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

const metalHawkesMarkovThreadCountGo = 256

func runMetalHawkesIntensity(
	events tensor.Tensor,
	queryTimes tensor.Tensor,
	baseline tensor.Tensor,
	alpha tensor.Tensor,
	beta tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalHawkesIntensity(events, queryTimes, baseline, alpha, beta, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.events, config.first, config.second, config.third, config.fourth)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_hawkes_intensity(
		config.events.bridge.device, C.int(config.elementDType), config.events.buffer, config.first.buffer,
		config.second.buffer, config.third.buffer, config.fourth.buffer, config.out.buffer,
		C.uint32_t(config.eventCount), C.uint32_t(config.outputCount), C.uint64_t(token), &status,
	)

	return finishMetalHawkesMarkovDispatch("hawkes_intensity", token, rc, status)
}

func runMetalHawkesKernelMatrix(
	events tensor.Tensor,
	alpha tensor.Tensor,
	beta tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalHawkesKernelMatrix(events, alpha, beta, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.events, config.first, config.second)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_hawkes_kernel_matrix(
		config.events.bridge.device, C.int(config.elementDType), config.events.buffer,
		config.first.buffer, config.second.buffer, config.out.buffer,
		C.uint32_t(config.eventCount), C.uint64_t(token), &status,
	)

	return finishMetalHawkesMarkovDispatch("hawkes_kernel_matrix", token, rc, status)
}

func runMetalHawkesLogLikelihood(
	events tensor.Tensor,
	totalTime tensor.Tensor,
	baseline tensor.Tensor,
	alpha tensor.Tensor,
	beta tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalHawkesLogLikelihood(events, totalTime, baseline, alpha, beta, out)
	if err != nil {
		return err
	}

	if err := config.allocateScratch(); err != nil {
		return err
	}

	token, err := config.beginCompletion(config.events, config.first, config.second, config.third, config.fourth)
	if err != nil {
		config.closeScratch()
		return err
	}

	config.closeScratch()
	status := C.MetalStatus{}
	rc := C.metal_dispatch_hawkes_log_likelihood(
		config.events.bridge.device, C.int(config.elementDType), config.events.buffer, config.first.buffer,
		config.second.buffer, config.third.buffer, config.fourth.buffer, config.scratch.buffer, config.out.buffer,
		C.uint32_t(config.eventCount), C.uint32_t(config.partialCount), C.uint64_t(token), &status,
	)

	return finishMetalHawkesMarkovDispatch("hawkes_log_likelihood", token, rc, status)
}

func runMetalMarkovMutualInformation(joint tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalMarkovMutualInformation(joint, out)
	if err != nil {
		return err
	}

	if err := config.allocateScratch(); err != nil {
		return err
	}

	token, err := config.beginCompletion(config.matrix)
	if err != nil {
		config.closeScratch()
		return err
	}

	config.closeScratch()
	status := C.MetalStatus{}
	rc := C.metal_dispatch_markov_mutual_information(
		config.matrix.bridge.device, C.int(config.elementDType), config.matrix.buffer, config.scratch.buffer,
		config.out.buffer, C.uint32_t(config.rows), C.uint32_t(config.cols),
		C.uint32_t(config.partialCount), C.uint64_t(token), &status,
	)

	return finishMetalHawkesMarkovDispatch("markov_mutual_information", token, rc, status)
}

func runMetalMarkovBlanketPartition(adjacency tensor.Tensor, internal tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalMarkovBlanketPartition(adjacency, internal, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.matrix, config.labels)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_markov_blanket_partition(
		config.matrix.bridge.device, C.int(config.elementDType), config.matrix.buffer,
		config.labels.buffer, config.out.buffer, C.uint32_t(config.rows),
		C.uint32_t(config.cols), C.uint64_t(token), &status,
	)

	return finishMetalHawkesMarkovDispatch("markov_blanket_partition", token, rc, status)
}

func runMetalMarkovFlow(mi tensor.Tensor, partition tensor.Tensor, out tensor.Tensor, targetLabel int32) error {
	config, err := requireMetalMarkovFlow(mi, partition, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.matrix, config.labels)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_markov_flow(
		config.matrix.bridge.device, C.int(config.elementDType), config.matrix.buffer,
		config.labels.buffer, config.out.buffer, C.uint32_t(config.rows),
		C.int32_t(targetLabel), C.uint64_t(token), &status,
	)

	return finishMetalHawkesMarkovDispatch("markov_flow", token, rc, status)
}

func metalHawkesMarkovPartialCount(elementCount int) int {
	return (elementCount + metalHawkesMarkovThreadCountGo - 1) / metalHawkesMarkovThreadCountGo
}

func newMetalHawkesMarkovScratch(bridge *metalBridge, partialCount int) (*metalTensor, error) {
	shape, err := tensor.NewShape([]int{partialCount})
	if err != nil {
		return nil, err
	}

	return bridge.empty(shape, dtype.Float32)
}

func (config *metalHawkesScalarConfig) allocateScratch() error {
	scratchElements := int(config.scratchCount)
	if scratchElements == 0 {
		scratchElements = int(config.partialCount)
	}

	scratch, err := newMetalHawkesMarkovScratch(config.out.bridge, scratchElements)
	if err != nil {
		return err
	}

	config.scratch = scratch
	return nil
}

func (config *metalMarkovMatrixConfig) allocateScratch() error {
	scratch, err := newMetalHawkesMarkovScratch(config.out.bridge, int(config.partialCount))
	if err != nil {
		return err
	}

	config.scratch = scratch
	return nil
}

func (config metalHawkesScalarConfig) beginCompletion(sources ...*metalTensor) (uint64, error) {
	return metalCompletions.BeginMany([]*metalTensor{config.out, config.scratch}, sources...)
}

func (config metalMarkovMatrixConfig) beginCompletion(sources ...*metalTensor) (uint64, error) {
	return metalCompletions.BeginMany([]*metalTensor{config.out, config.scratch}, sources...)
}

func (config metalHawkesScalarConfig) closeScratch() {
	_ = config.scratch.Close()
}

func (config metalMarkovMatrixConfig) closeScratch() {
	_ = config.scratch.Close()
}

func finishMetalHawkesMarkovDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
