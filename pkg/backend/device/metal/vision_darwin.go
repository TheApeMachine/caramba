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

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type metalConv2DConfig struct {
	input        *metalTensor
	weight       *metalTensor
	bias         *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	batch        uint32
	inChannels   uint32
	inHeight     uint32
	inWidth      uint32
	outChannels  uint32
	kernelHeight uint32
	kernelWidth  uint32
	outHeight    uint32
	outWidth     uint32
}

type metalPool2DConfig struct {
	input        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	batch        uint32
	channels     uint32
	inHeight     uint32
	inWidth      uint32
	outHeight    uint32
	outWidth     uint32
}

func runMetalConv2D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalConv2D(input, weight, bias, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.weight, config.bias)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_conv2d(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.weight.buffer,
		config.bias.buffer,
		config.out.buffer,
		C.uint32_t(config.batch),
		C.uint32_t(config.inChannels),
		C.uint32_t(config.inHeight),
		C.uint32_t(config.inWidth),
		C.uint32_t(config.outChannels),
		C.uint32_t(config.kernelHeight),
		C.uint32_t(config.kernelWidth),
		C.uint32_t(config.outHeight),
		C.uint32_t(config.outWidth),
		C.uint64_t(token),
		&status,
	)

	return finishMetalVisionDispatch("conv2d", token, rc, status)
}

func runMetalMaxPool2D(input tensor.Tensor, out tensor.Tensor) error {
	return runMetalPool2D("max_pool2d", input, out, true, false)
}

func runMetalAvgPool2D(input tensor.Tensor, out tensor.Tensor) error {
	return runMetalPool2D("avg_pool2d", input, out, false, false)
}

func runMetalAdaptiveAvgPool2D(input tensor.Tensor, out tensor.Tensor) error {
	return runMetalPool2D("adaptive_avg_pool2d", input, out, false, true)
}

func runMetalAdaptiveMaxPool2D(input tensor.Tensor, out tensor.Tensor) error {
	return runMetalPool2D("adaptive_max_pool2d", input, out, true, true)
}

func runMetalPool2D(
	name string,
	input tensor.Tensor,
	out tensor.Tensor,
	useMax bool,
	adaptive bool,
) error {
	config, err := requireMetalPool2D(input, out)
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
	rc := C.metal_dispatch_pool2d(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.out.buffer,
		C.uint32_t(config.batch),
		C.uint32_t(config.channels),
		C.uint32_t(config.inHeight),
		C.uint32_t(config.inWidth),
		C.uint32_t(config.outHeight),
		C.uint32_t(config.outWidth),
		C.bool(useMax),
		C.bool(adaptive),
		C.uint64_t(token),
		&status,
	)

	return finishMetalVisionDispatch(name, token, rc, status)
}

func requireMetalConv2D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalConv2DConfig, error) {
	inputTensor, weightTensor, biasTensor, outTensor, err :=
		requireMetalTensors4(input, weight, bias, out)
	if err != nil {
		return metalConv2DConfig{}, err
	}

	if err := requireMetalVisionSameDTypeAndBridge(inputTensor, weightTensor, biasTensor, outTensor); err != nil {
		return metalConv2DConfig{}, err
	}

	config, err := metalConv2DDims(inputTensor, weightTensor, biasTensor, outTensor)
	if err != nil {
		return metalConv2DConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalConv2DConfig{}, err
	}

	config.input = inputTensor
	config.weight = weightTensor
	config.bias = biasTensor
	config.out = outTensor
	config.elementDType = elementDType
	return config, nil
}

func metalConv2DDims(
	input *metalTensor,
	weight *metalTensor,
	bias *metalTensor,
	out *metalTensor,
) (metalConv2DConfig, error) {
	inputDims := input.shape.Dims()
	weightDims := weight.shape.Dims()
	biasDims := bias.shape.Dims()
	outDims := out.shape.Dims()

	if len(inputDims) != 4 || len(weightDims) != 4 ||
		len(biasDims) != 1 || len(outDims) != 4 {
		return metalConv2DConfig{}, tensor.ErrShapeMismatch
	}

	if weightDims[1] != inputDims[1] || biasDims[0] != weightDims[0] ||
		outDims[0] != inputDims[0] || outDims[1] != weightDims[0] {
		return metalConv2DConfig{}, tensor.ErrShapeMismatch
	}

	if out.shape.Len() > math.MaxUint32 {
		return metalConv2DConfig{}, tensor.ErrShapeMismatch
	}

	return metalConv2DConfig{
		batch:        uint32(inputDims[0]),
		inChannels:   uint32(inputDims[1]),
		inHeight:     uint32(inputDims[2]),
		inWidth:      uint32(inputDims[3]),
		outChannels:  uint32(weightDims[0]),
		kernelHeight: uint32(weightDims[2]),
		kernelWidth:  uint32(weightDims[3]),
		outHeight:    uint32(outDims[2]),
		outWidth:     uint32(outDims[3]),
	}, nil
}

func requireMetalPool2D(input tensor.Tensor, out tensor.Tensor) (metalPool2DConfig, error) {
	inputTensor, outTensor, err := requireUnaryElementwiseTensors(input, out)
	if err != nil && !errors.Is(err, tensor.ErrShapeMismatch) {
		return metalPool2DConfig{}, err
	}

	if err != nil {
		inputTensor, outTensor, err = metalPool2DTensors(input, out)
		if err != nil {
			return metalPool2DConfig{}, err
		}
	}

	config, err := metalPool2DDims(inputTensor, outTensor)
	if err != nil {
		return metalPool2DConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalPool2DConfig{}, err
	}

	config.input = inputTensor
	config.out = outTensor
	config.elementDType = elementDType
	return config, nil
}

func metalPool2DTensors(input tensor.Tensor, out tensor.Tensor) (*metalTensor, *metalTensor, error) {
	inputTensor, err := requireMetalTensor(input)
	if err != nil {
		return nil, nil, err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return nil, nil, err
	}

	if inputTensor.dtype != outTensor.dtype {
		return nil, nil, tensor.ErrDTypeMismatch
	}

	if inputTensor.bridge != outTensor.bridge {
		return nil, nil, errors.New("metal vision: tensors belong to different Metal backends")
	}

	return inputTensor, outTensor, nil
}

func metalPool2DDims(input *metalTensor, out *metalTensor) (metalPool2DConfig, error) {
	inputDims := input.shape.Dims()
	outDims := out.shape.Dims()

	if len(inputDims) != 4 || len(outDims) != 4 {
		return metalPool2DConfig{}, tensor.ErrShapeMismatch
	}

	if outDims[0] != inputDims[0] || outDims[1] != inputDims[1] ||
		out.shape.Len() > math.MaxUint32 {
		return metalPool2DConfig{}, tensor.ErrShapeMismatch
	}

	return metalPool2DConfig{
		batch:     uint32(inputDims[0]),
		channels:  uint32(inputDims[1]),
		inHeight:  uint32(inputDims[2]),
		inWidth:   uint32(inputDims[3]),
		outHeight: uint32(outDims[2]),
		outWidth:  uint32(outDims[3]),
	}, nil
}

func requireMetalVisionSameDTypeAndBridge(tensors ...*metalTensor) error {
	if len(tensors) == 0 {
		return tensor.ErrShapeMismatch
	}

	storageDType := tensors[0].dtype
	bridge := tensors[0].bridge

	for _, target := range tensors[1:] {
		if target.dtype != storageDType {
			return tensor.ErrDTypeMismatch
		}

		if target.bridge != bridge {
			return errors.New("metal vision: tensors belong to different Metal backends")
		}
	}

	return nil
}

func finishMetalVisionDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
