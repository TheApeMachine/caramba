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
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type metalAttentionConfig struct {
	query        *metalTensor
	key          *metalTensor
	value        *metalTensor
	scores       *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	seqQ         uint32
	seqK         uint32
	depth        uint32
	valueDim     uint32
}

type metalRoPEConfig struct {
	input        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	seqLen       uint32
	numHeads     uint32
	headDim      uint32
	pairCount    uint32
}

func runMetalAttention(
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalAttention(query, key, value, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	config.scores, err = newMetalAttentionScores(config)
	if err != nil {
		return err
	}
	defer func() {
		_ = config.scores.Close()
	}()

	token, err := metalCompletions.BeginMany(
		[]*metalTensor{config.out, config.scores},
		config.query,
		config.key,
		config.value,
	)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_attention(
		config.query.bridge.device,
		C.int(config.elementDType),
		config.query.buffer,
		config.key.buffer,
		config.value.buffer,
		config.scores.buffer,
		config.out.buffer,
		C.uint32_t(config.seqQ),
		C.uint32_t(config.seqK),
		C.uint32_t(config.depth),
		C.uint32_t(config.valueDim),
		C.uint64_t(token),
		&status,
	)

	return finishMetalTransformerDispatch("attention", token, rc, status)
}

func runMetalRoPE(input tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalRoPE(input, out)
	if err != nil {
		return err
	}

	if config.pairCount == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_rope(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.out.buffer,
		C.uint32_t(config.seqLen),
		C.uint32_t(config.numHeads),
		C.uint32_t(config.headDim),
		C.uint32_t(config.pairCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalTransformerDispatch("rope", token, rc, status)
}

func requireMetalAttention(
	query tensor.Tensor,
	key tensor.Tensor,
	value tensor.Tensor,
	out tensor.Tensor,
) (metalAttentionConfig, error) {
	tensors, err := requireMetalTensors(query, key, value, out)
	if err != nil {
		return metalAttentionConfig{}, err
	}

	config := metalAttentionConfig{
		query: tensors[0], key: tensors[1], value: tensors[2], out: tensors[3],
	}
	if err := requireMetalAttentionSameDTypeAndBridge(config); err != nil {
		return metalAttentionConfig{}, err
	}

	if err := requireMetalAttentionDims(&config); err != nil {
		return metalAttentionConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.query.dtype)
	if err != nil {
		return metalAttentionConfig{}, err
	}

	config.elementDType = elementDType
	return config, nil
}

func requireMetalRoPE(input tensor.Tensor, out tensor.Tensor) (metalRoPEConfig, error) {
	inputTensor, outTensor, err := requireMetalMathSameDType(input, out)
	if err != nil {
		return metalRoPEConfig{}, err
	}

	seqLen, numHeads, headDim, pairCount, err := metalRoPEDims(inputTensor, outTensor)
	if err != nil {
		return metalRoPEConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalRoPEConfig{}, err
	}

	return metalRoPEConfig{
		input:        inputTensor,
		out:          outTensor,
		elementDType: elementDType,
		seqLen:       uint32(seqLen),
		numHeads:     uint32(numHeads),
		headDim:      uint32(headDim),
		pairCount:    uint32(pairCount),
	}, nil
}

func requireMetalAttentionSameDTypeAndBridge(config metalAttentionConfig) error {
	if config.query.dtype != config.key.dtype ||
		config.query.dtype != config.value.dtype ||
		config.query.dtype != config.out.dtype {
		return tensor.ErrDTypeMismatch
	}

	if config.query.bridge != config.key.bridge ||
		config.query.bridge != config.value.bridge ||
		config.query.bridge != config.out.bridge {
		return errors.New("metal attention: tensors belong to different Metal backends")
	}

	return nil
}

func requireMetalAttentionDims(config *metalAttentionConfig) error {
	queryDims := config.query.shape.Dims()
	keyDims := config.key.shape.Dims()
	valueDims := config.value.shape.Dims()
	outDims := config.out.shape.Dims()

	if len(queryDims) != 2 || len(keyDims) != 2 || len(valueDims) != 2 || len(outDims) != 2 {
		return tensor.ErrShapeMismatch
	}

	if keyDims[1] != queryDims[1] || valueDims[0] != keyDims[0] {
		return tensor.ErrShapeMismatch
	}

	if outDims[0] != queryDims[0] || outDims[1] != valueDims[1] {
		return tensor.ErrShapeMismatch
	}

	if queryDims[1] <= 0 || keyDims[0] <= 0 || valueDims[1] <= 0 {
		return tensor.ErrShapeMismatch
	}

	if err := requireTransformerUint32(queryDims[0], keyDims[0], queryDims[1], valueDims[1]); err != nil {
		return err
	}

	config.seqQ = uint32(queryDims[0])
	config.seqK = uint32(keyDims[0])
	config.depth = uint32(queryDims[1])
	config.valueDim = uint32(valueDims[1])
	return nil
}

func newMetalAttentionScores(config metalAttentionConfig) (*metalTensor, error) {
	scoreShape, err := tensor.NewShape([]int{int(config.seqQ), int(config.seqK)})
	if err != nil {
		return nil, err
	}

	return config.query.bridge.empty(scoreShape, dtype.Float32)
}

func metalRoPEDims(input *metalTensor, out *metalTensor) (int, int, int, int, error) {
	dims := input.shape.Dims()
	if len(dims) != 3 || !input.shape.Equal(out.shape) {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	seqLen, numHeads, headDim := dims[0], dims[1], dims[2]
	if headDim%2 != 0 {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	if seqLen < 0 || numHeads < 0 || headDim < 0 {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	pairCount64 := int64(seqLen) * int64(numHeads) * int64(headDim/2)
	if pairCount64 > math.MaxUint32 {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	return seqLen, numHeads, headDim, int(pairCount64), nil
}
