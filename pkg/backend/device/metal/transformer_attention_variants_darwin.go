//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"errors"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type metalMultiHeadAttentionVariant int

const (
	metalMultiHeadAttentionVariantFull metalMultiHeadAttentionVariant = iota
	metalMultiHeadAttentionVariantGrouped
	metalMultiHeadAttentionVariantSliding
)

type metalMultiHeadAttentionConfig struct {
	query        *metalTensor
	key          *metalTensor
	value        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	variant      metalMultiHeadAttentionVariant
	seqQ         uint32
	seqK         uint32
	numHeads     uint32
	kvHeads      uint32
	headDim      uint32
	windowSize   uint32
	causal       uint32
}

func runMetalMultiHeadAttention(
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
	out tensor.Tensor,
) error {
	return runMetalMultiHeadAttentionVariant(
		metalMultiHeadAttentionVariantFull, query, key, value, out,
	)
}

func runMetalGroupedQueryAttention(
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
	out tensor.Tensor,
) error {
	return runMetalMultiHeadAttentionVariant(
		metalMultiHeadAttentionVariantGrouped, query, key, value, out,
	)
}

func runMetalSlidingWindowAttention(
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
	out tensor.Tensor,
) error {
	return runMetalMultiHeadAttentionVariant(
		metalMultiHeadAttentionVariantSliding, query, key, value, out,
	)
}

func runMetalMultiHeadAttentionVariant(
	variant metalMultiHeadAttentionVariant,
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalMultiHeadAttention(variant, query, key, value, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.query, config.key, config.value)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_multi_head_attention(
		config.query.bridge.device,
		C.int(config.elementDType),
		C.int(config.variant),
		config.query.buffer,
		config.key.buffer,
		config.value.buffer,
		config.out.buffer,
		C.uint32_t(config.seqQ),
		C.uint32_t(config.seqK),
		C.uint32_t(config.numHeads),
		C.uint32_t(config.kvHeads),
		C.uint32_t(config.headDim),
		C.uint32_t(config.windowSize),
		C.uint32_t(config.causal),
		C.uint64_t(token),
		&status,
	)

	return finishMetalTransformerDispatch(metalMultiHeadAttentionName(variant), token, rc, status)
}

func requireMetalMultiHeadAttention(
	variant metalMultiHeadAttentionVariant,
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
	out tensor.Tensor,
) (metalMultiHeadAttentionConfig, error) {
	tensors, err := requireMetalTensors(query, key, value, out)
	if err != nil {
		return metalMultiHeadAttentionConfig{}, err
	}

	config := metalMultiHeadAttentionConfig{
		query: tensors[0], key: tensors[1], value: tensors[2], out: tensors[3],
		variant: variant, numHeads: 8, kvHeads: 8,
	}
	config.applyVariant()

	if err := requireMetalMultiHeadAttentionSameDTypeAndBridge(config); err != nil {
		return metalMultiHeadAttentionConfig{}, err
	}

	if err := requireMetalMultiHeadAttentionDims(&config); err != nil {
		return metalMultiHeadAttentionConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.query.dtype)
	if err != nil {
		return metalMultiHeadAttentionConfig{}, err
	}

	config.elementDType = elementDType
	return config, nil
}

func requireMetalMultiHeadAttentionSameDTypeAndBridge(
	config metalMultiHeadAttentionConfig,
) error {
	if config.query.dtype != config.key.dtype ||
		config.query.dtype != config.value.dtype ||
		config.query.dtype != config.out.dtype {
		return tensor.ErrDTypeMismatch
	}

	if config.query.bridge != config.key.bridge ||
		config.query.bridge != config.value.bridge ||
		config.query.bridge != config.out.bridge {
		return errors.New("metal multi-head attention: tensors belong to different Metal backends")
	}

	return nil
}

func requireMetalMultiHeadAttentionDims(config *metalMultiHeadAttentionConfig) error {
	queryDims := config.query.shape.Dims()
	keyDims := config.key.shape.Dims()
	valueDims := config.value.shape.Dims()
	outDims := config.out.shape.Dims()

	if len(queryDims) != 2 || len(keyDims) != 2 || len(valueDims) != 2 || len(outDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	if queryDims[1]%int(config.numHeads) != 0 {
		return tensor.ErrShapeMismatch
	}

	headDim := queryDims[1] / int(config.numHeads)
	if headDim <= 0 || keyDims[1] != int(config.kvHeads)*headDim {
		return tensor.ErrShapeMismatch
	}

	if valueDims[0] != keyDims[0] || valueDims[1] != keyDims[1] {
		return tensor.ErrShapeMismatch
	}

	if outDims[0] != queryDims[0] || outDims[1] != queryDims[1] {
		return tensor.ErrShapeMismatch
	}

	if err := requireTransformerUint32(
		queryDims[0], keyDims[0], int(config.numHeads), int(config.kvHeads), headDim,
	); err != nil {
		return err
	}

	config.seqQ = uint32(queryDims[0])
	config.seqK = uint32(keyDims[0])
	config.headDim = uint32(headDim)
	return nil
}

func (config *metalMultiHeadAttentionConfig) applyVariant() {
	switch config.variant {
	case metalMultiHeadAttentionVariantGrouped:
		config.kvHeads = config.numHeads / 4
	case metalMultiHeadAttentionVariantSliding:
		config.causal = 1
		config.windowSize = 128
	}
}

func metalMultiHeadAttentionName(variant metalMultiHeadAttentionVariant) string {
	switch variant {
	case metalMultiHeadAttentionVariantGrouped:
		return "grouped_query_attention"
	case metalMultiHeadAttentionVariantSliding:
		return "sliding_window_attention"
	default:
		return "multi_head_attention"
	}
}
