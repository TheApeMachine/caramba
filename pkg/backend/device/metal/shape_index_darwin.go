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
	"runtime"
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

type metalShapeIndexedConfig struct {
	first        *metalTensor
	second       *metalTensor
	third        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	rows         uint32
	inner        uint32
	cols         uint32
	count        uint32
}

type metalShapeTransposeConfig struct {
	input        *metalTensor
	permutation  *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	rank         uint32
	count        uint32
	permutationV []uint32
	inputStrides []uint32
	outStrides   []uint32
}

func runMetalGather(source tensor.Tensor, indices tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalGather(source, indices, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.first, config.second)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_gather(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.out.buffer, C.uint32_t(config.rows),
		C.uint32_t(config.inner), C.uint32_t(config.cols), C.uint64_t(token), &status,
	)

	return finishMetalShapeIndexDispatch("gather", token, rc, status)
}

func runMetalScatter(
	target tensor.Tensor,
	indices tensor.Tensor,
	updates tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalScatter(target, indices, updates, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.first, config.second, config.third)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_scatter(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.third.buffer, config.out.buffer, C.uint32_t(config.rows),
		C.uint32_t(config.inner), C.uint32_t(config.cols), C.uint64_t(token), &status,
	)

	return finishMetalShapeIndexDispatch("scatter", token, rc, status)
}

func runMetalWhere(
	mask tensor.Tensor,
	positive tensor.Tensor,
	negative tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalWhere(mask, positive, negative, out)
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
	rc := C.metal_dispatch_where(
		config.first.bridge.device, C.int(config.elementDType), config.first.buffer,
		config.second.buffer, config.third.buffer, config.out.buffer, C.uint32_t(config.count),
		C.uint64_t(token), &status,
	)

	return finishMetalShapeIndexDispatch("where", token, rc, status)
}

func runMetalMaskedFill(
	input tensor.Tensor,
	mask tensor.Tensor,
	scalar tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalMaskedFill(input, mask, scalar, out)
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
	rc := C.metal_dispatch_masked_fill(
		config.first.bridge.device, C.int(config.elementDType), config.second.buffer,
		config.first.buffer, config.third.buffer, config.out.buffer, C.uint32_t(config.count),
		C.uint64_t(token), &status,
	)

	return finishMetalShapeIndexDispatch("masked_fill", token, rc, status)
}

func runMetalTranspose(input tensor.Tensor, permutation tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalTranspose(input, permutation, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.permutation)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_transpose(
		config.input.bridge.device, C.int(config.elementDType), config.input.buffer,
		config.out.buffer, C.uint32_t(config.rank), C.uint32_t(config.count),
		cUint32Pointer(config.permutationV), cUint32Pointer(config.inputStrides),
		cUint32Pointer(config.outStrides), C.uint64_t(token), &status,
	)
	runtime.KeepAlive(config.permutationV)
	runtime.KeepAlive(config.inputStrides)
	runtime.KeepAlive(config.outStrides)

	return finishMetalShapeIndexDispatch("transpose", token, rc, status)
}

func cUint32Pointer(values []uint32) *C.uint32_t {
	if len(values) == 0 {
		return nil
	}

	return (*C.uint32_t)(unsafe.Pointer(&values[0]))
}

func finishMetalShapeIndexDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
