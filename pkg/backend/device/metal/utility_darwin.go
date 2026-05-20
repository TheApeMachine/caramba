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
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type metalUtilityTensorPair struct {
	input *metalTensor
	out   *metalTensor
	count uint32
}

type metalWeightFreezeMaskConfig struct {
	mask         *metalTensor
	gradients    *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
}

func runMetalCheckpointEncodeFloat32(input tensor.Tensor, out tensor.Tensor) error {
	inputTensor, outTensor, err := requireMetalCheckpointEncode(input, out)
	if err != nil {
		return err
	}

	dims := uint64Dims(inputTensor.shape.Dims())
	token, err := metalCompletions.Begin(outTensor, inputTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_checkpoint_encode_float32(
		inputTensor.bridge.device,
		inputTensor.buffer,
		outTensor.buffer,
		C.uint32_t(len(dims)),
		C.uint32_t(inputTensor.shape.Len()),
		cUint64Pointer(dims),
		C.uint64_t(token),
		&status,
	)
	runtime.KeepAlive(dims)

	return finishMetalUtilityDispatch("checkpoint_encode_float32", token, rc, status)
}

func runMetalCheckpointDecodeFloat32(input tensor.Tensor, out tensor.Tensor) error {
	config, headerBytes, err := requireMetalCheckpointDecode(input, out)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.out, config.input)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_checkpoint_decode_float32(
		config.input.bridge.device,
		config.input.buffer,
		config.out.buffer,
		C.uint32_t(headerBytes),
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalUtilityDispatch("checkpoint_decode_float32", token, rc, status)
}

func runMetalTokenizerPackInt32(input tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalUtilityPair(input, out, dtype.Int32, dtype.Int32)
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
	rc := C.metal_dispatch_tokenizer_pack_int32(
		config.input.bridge.device,
		config.input.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalUtilityDispatch("tokenizer_pack_int32", token, rc, status)
}

func runMetalWeightFreezeMask(mask tensor.Tensor, gradients tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalWeightFreezeMask(mask, gradients, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.mask, config.gradients)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_weight_freeze_mask(
		config.mask.bridge.device,
		C.int(config.elementDType),
		config.mask.buffer,
		config.gradients.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalUtilityDispatch("weight_freeze_mask", token, rc, status)
}

func requireMetalCheckpointEncode(
	input tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	inputTensor, outTensor, err := requireMetalUtilityTensors(input, out)
	if err != nil {
		return nil, nil, err
	}

	if inputTensor.dtype != dtype.Float32 || outTensor.dtype != dtype.Uint8 {
		return nil, nil, tensor.ErrDTypeMismatch
	}

	headerBytes := 16 + len(inputTensor.shape.Dims())*8
	dataBytes := inputTensor.shape.Len() * 4
	if outTensor.shape.Len() != headerBytes+dataBytes {
		return nil, nil, tensor.ErrShapeMismatch
	}

	if inputTensor.shape.Len() > math.MaxUint32 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return inputTensor, outTensor, nil
}

func requireMetalCheckpointDecode(
	input tensor.Tensor,
	out tensor.Tensor,
) (metalUtilityTensorPair, int, error) {
	inputTensor, outTensor, err := requireMetalUtilityTensors(input, out)
	if err != nil {
		return metalUtilityTensorPair{}, 0, err
	}

	if inputTensor.dtype != dtype.Uint8 || outTensor.dtype != dtype.Float32 {
		return metalUtilityTensorPair{}, 0, tensor.ErrDTypeMismatch
	}

	if outTensor.shape.Len() > math.MaxUint32 {
		return metalUtilityTensorPair{}, 0, tensor.ErrShapeMismatch
	}

	headerBytes := inputTensor.shape.Len() - outTensor.shape.Len()*4
	if headerBytes < 16 || (headerBytes-16)%8 != 0 {
		return metalUtilityTensorPair{}, 0, tensor.ErrShapeMismatch
	}

	return metalUtilityTensorPair{
		input: inputTensor,
		out:   outTensor,
		count: uint32(outTensor.shape.Len()),
	}, headerBytes, nil
}

func requireMetalUtilityPair(
	input tensor.Tensor,
	out tensor.Tensor,
	inputDType dtype.DType,
	outDType dtype.DType,
) (metalUtilityTensorPair, error) {
	inputTensor, outTensor, err := requireMetalUtilityTensors(input, out)
	if err != nil {
		return metalUtilityTensorPair{}, err
	}

	if inputTensor.dtype != inputDType || outTensor.dtype != outDType {
		return metalUtilityTensorPair{}, tensor.ErrDTypeMismatch
	}

	if !inputTensor.shape.Equal(outTensor.shape) || inputTensor.shape.Len() > math.MaxUint32 {
		return metalUtilityTensorPair{}, tensor.ErrShapeMismatch
	}

	return metalUtilityTensorPair{
		input: inputTensor,
		out:   outTensor,
		count: uint32(inputTensor.shape.Len()),
	}, nil
}

func requireMetalUtilityTensors(
	input tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(input, out)
	if err != nil {
		return nil, nil, err
	}

	if tensors[0].bridge != tensors[1].bridge {
		return nil, nil, errors.New("metal utility: tensors belong to different Metal backends")
	}

	return tensors[0], tensors[1], nil
}

func requireMetalWeightFreezeMask(
	mask tensor.Tensor,
	gradients tensor.Tensor,
	out tensor.Tensor,
) (metalWeightFreezeMaskConfig, error) {
	tensors, err := requireMetalTensors(mask, gradients, out)
	if err != nil {
		return metalWeightFreezeMaskConfig{}, err
	}

	if err := requireMetalWeightFreezeMaskShapes(tensors[0], tensors[1], tensors[2]); err != nil {
		return metalWeightFreezeMaskConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[1].dtype)
	if err != nil {
		return metalWeightFreezeMaskConfig{}, err
	}

	return metalWeightFreezeMaskConfig{
		mask:         tensors[0],
		gradients:    tensors[1],
		out:          tensors[2],
		elementDType: elementDType,
		count:        uint32(tensors[1].shape.Len()),
	}, nil
}

func requireMetalWeightFreezeMaskShapes(
	mask *metalTensor,
	gradients *metalTensor,
	out *metalTensor,
) error {
	if mask.bridge != gradients.bridge || mask.bridge != out.bridge {
		return errors.New("metal utility: tensors belong to different Metal backends")
	}

	if mask.dtype != dtype.Bool || gradients.dtype != out.dtype {
		return tensor.ErrDTypeMismatch
	}

	if !mask.shape.Equal(gradients.shape) || !gradients.shape.Equal(out.shape) {
		return tensor.ErrShapeMismatch
	}

	if gradients.shape.Len() > math.MaxUint32 {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func uint64Dims(dims []int) []uint64 {
	encoded := make([]uint64, len(dims))

	for index, dim := range dims {
		encoded[index] = uint64(dim)
	}

	return encoded
}

func cUint64Pointer(values []uint64) *C.uint64_t {
	if len(values) == 0 {
		return nil
	}

	return (*C.uint64_t)(unsafe.Pointer(&values[0]))
}

func finishMetalUtilityDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
