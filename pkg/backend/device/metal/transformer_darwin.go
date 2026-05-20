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

func runMetalEmbeddingLookup(
	table tensor.Tensor,
	indices tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalEmbeddingLookup(table, indices, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.table, config.indices)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_embedding_lookup(
		config.table.bridge.device,
		C.int(config.elementDType),
		config.table.buffer,
		config.indices.buffer,
		config.out.buffer,
		C.uint32_t(config.vocab),
		C.uint32_t(config.hidden),
		C.uint32_t(config.indexCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalTransformerDispatch("embedding_lookup", token, rc, status)
}

func runMetalEmbeddingBag(
	table tensor.Tensor,
	indices tensor.Tensor,
	offsets tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalEmbeddingBag(table, indices, offsets, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.table, config.indices, config.offsets)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_embedding_bag(
		config.table.bridge.device,
		C.int(config.elementDType),
		config.table.buffer,
		config.indices.buffer,
		config.offsets.buffer,
		config.out.buffer,
		C.uint32_t(config.vocab),
		C.uint32_t(config.hidden),
		C.uint32_t(config.indexCount),
		C.uint32_t(config.bagCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalTransformerDispatch("embedding_bag", token, rc, status)
}

func runMetalApplyMask(
	input tensor.Tensor,
	mask tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalApplyMask(input, mask, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.mask)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_apply_mask(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.mask.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalTransformerDispatch("apply_mask", token, rc, status)
}

func runMetalCausalMask(input tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalCausalMask(input, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_causal_mask(
		config.out.bridge.device,
		C.int(config.elementDType),
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	return finishMetalTransformerDispatch("causal_mask", token, rc, status)
}

func runMetalALiBiBias(
	scores tensor.Tensor,
	slope tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalALiBiBias(scores, slope, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.scores, config.slope)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_alibi_bias(
		config.scores.bridge.device,
		C.int(config.elementDType),
		config.scores.buffer,
		config.slope.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	return finishMetalTransformerDispatch("alibi_bias", token, rc, status)
}

func finishMetalTransformerDispatch(
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
