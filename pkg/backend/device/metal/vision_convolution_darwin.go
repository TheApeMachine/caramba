//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

type metalConv1DConfig struct {
	input        *metalTensor
	weight       *metalTensor
	bias         *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	batch        uint32
	inChannels   uint32
	inLength     uint32
	outChannels  uint32
	kernelLength uint32
	outLength    uint32
}

type metalConv3DConfig struct {
	input        *metalTensor
	weight       *metalTensor
	bias         *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	batch        uint32
	inChannels   uint32
	inDepth      uint32
	inHeight     uint32
	inWidth      uint32
	outChannels  uint32
	kernelDepth  uint32
	kernelHeight uint32
	kernelWidth  uint32
	outDepth     uint32
	outHeight    uint32
	outWidth     uint32
}

func runMetalConv1D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalConv1D(input, weight, bias, out)
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
	rc := C.metal_dispatch_conv1d(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.weight.buffer,
		config.bias.buffer,
		config.out.buffer,
		C.uint32_t(config.batch),
		C.uint32_t(config.inChannels),
		C.uint32_t(config.inLength),
		C.uint32_t(config.outChannels),
		C.uint32_t(config.kernelLength),
		C.uint32_t(config.outLength),
		C.uint64_t(token),
		&status,
	)

	return finishMetalVisionDispatch("conv1d", token, rc, status)
}

func runMetalConv3D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalConv3D(input, weight, bias, out)
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
	rc := C.metal_dispatch_conv3d(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.weight.buffer,
		config.bias.buffer,
		config.out.buffer,
		C.uint32_t(config.batch),
		C.uint32_t(config.inChannels),
		C.uint32_t(config.inDepth),
		C.uint32_t(config.inHeight),
		C.uint32_t(config.inWidth),
		C.uint32_t(config.outChannels),
		C.uint32_t(config.kernelDepth),
		C.uint32_t(config.kernelHeight),
		C.uint32_t(config.kernelWidth),
		C.uint32_t(config.outDepth),
		C.uint32_t(config.outHeight),
		C.uint32_t(config.outWidth),
		C.uint64_t(token),
		&status,
	)

	return finishMetalVisionDispatch("conv3d", token, rc, status)
}

func runMetalConvTranspose2D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalConvTranspose2D(input, weight, bias, out)
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
	rc := C.metal_dispatch_conv_transpose2d(
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

	return finishMetalVisionDispatch("conv_transpose2d", token, rc, status)
}

func requireMetalConv1D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalConv1DConfig, error) {
	tensors, err := requireMetalTensors(input, weight, bias, out)
	if err != nil {
		return metalConv1DConfig{}, err
	}

	if err := requireMetalVisionSameDTypeAndBridge(tensors...); err != nil {
		return metalConv1DConfig{}, err
	}

	config, err := metalConv1DDims(tensors[0], tensors[1], tensors[2], tensors[3])
	if err != nil {
		return metalConv1DConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalConv1DConfig{}, err
	}

	config.input = tensors[0]
	config.weight = tensors[1]
	config.bias = tensors[2]
	config.out = tensors[3]
	config.elementDType = elementDType
	return config, nil
}

func requireMetalConv3D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalConv3DConfig, error) {
	tensors, err := requireMetalTensors(input, weight, bias, out)
	if err != nil {
		return metalConv3DConfig{}, err
	}

	if err := requireMetalVisionSameDTypeAndBridge(tensors...); err != nil {
		return metalConv3DConfig{}, err
	}

	config, err := metalConv3DDims(tensors[0], tensors[1], tensors[2], tensors[3])
	if err != nil {
		return metalConv3DConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalConv3DConfig{}, err
	}

	config.input = tensors[0]
	config.weight = tensors[1]
	config.bias = tensors[2]
	config.out = tensors[3]
	config.elementDType = elementDType
	return config, nil
}

func requireMetalConvTranspose2D(
	input tensor.Tensor,
	weight tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalConv2DConfig, error) {
	tensors, err := requireMetalTensors(input, weight, bias, out)
	if err != nil {
		return metalConv2DConfig{}, err
	}

	if err := requireMetalVisionSameDTypeAndBridge(tensors...); err != nil {
		return metalConv2DConfig{}, err
	}

	config, err := metalConvTranspose2DDims(tensors[0], tensors[1], tensors[2], tensors[3])
	if err != nil {
		return metalConv2DConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalConv2DConfig{}, err
	}

	config.input = tensors[0]
	config.weight = tensors[1]
	config.bias = tensors[2]
	config.out = tensors[3]
	config.elementDType = elementDType
	return config, nil
}

func metalConv1DDims(
	input *metalTensor,
	weight *metalTensor,
	bias *metalTensor,
	out *metalTensor,
) (metalConv1DConfig, error) {
	inputDims := input.shape.Dims()
	weightDims := weight.shape.Dims()
	biasDims := bias.shape.Dims()
	outDims := out.shape.Dims()

	if len(inputDims) != 3 || len(weightDims) != 3 ||
		len(biasDims) != 1 || len(outDims) != 3 {
		return metalConv1DConfig{}, tensor.ErrShapeMismatch
	}

	if weightDims[1] != inputDims[1] || biasDims[0] != weightDims[0] ||
		outDims[0] != inputDims[0] || outDims[1] != weightDims[0] ||
		out.shape.Len() > math.MaxUint32 {
		return metalConv1DConfig{}, tensor.ErrShapeMismatch
	}

	return metalConv1DConfig{
		batch:        uint32(inputDims[0]),
		inChannels:   uint32(inputDims[1]),
		inLength:     uint32(inputDims[2]),
		outChannels:  uint32(weightDims[0]),
		kernelLength: uint32(weightDims[2]),
		outLength:    uint32(outDims[2]),
	}, nil
}

func metalConv3DDims(
	input *metalTensor,
	weight *metalTensor,
	bias *metalTensor,
	out *metalTensor,
) (metalConv3DConfig, error) {
	inputDims := input.shape.Dims()
	weightDims := weight.shape.Dims()
	biasDims := bias.shape.Dims()
	outDims := out.shape.Dims()

	if len(inputDims) != 5 || len(weightDims) != 5 ||
		len(biasDims) != 1 || len(outDims) != 5 {
		return metalConv3DConfig{}, tensor.ErrShapeMismatch
	}

	if weightDims[1] != inputDims[1] || biasDims[0] != weightDims[0] ||
		outDims[0] != inputDims[0] || outDims[1] != weightDims[0] ||
		out.shape.Len() > math.MaxUint32 {
		return metalConv3DConfig{}, tensor.ErrShapeMismatch
	}

	return metalConv3DConfig{
		batch:        uint32(inputDims[0]),
		inChannels:   uint32(inputDims[1]),
		inDepth:      uint32(inputDims[2]),
		inHeight:     uint32(inputDims[3]),
		inWidth:      uint32(inputDims[4]),
		outChannels:  uint32(weightDims[0]),
		kernelDepth:  uint32(weightDims[2]),
		kernelHeight: uint32(weightDims[3]),
		kernelWidth:  uint32(weightDims[4]),
		outDepth:     uint32(outDims[2]),
		outHeight:    uint32(outDims[3]),
		outWidth:     uint32(outDims[4]),
	}, nil
}

func metalConvTranspose2DDims(
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

	if weightDims[0] != inputDims[1] || biasDims[0] != weightDims[1] ||
		outDims[0] != inputDims[0] || outDims[1] != weightDims[1] ||
		out.shape.Len() > math.MaxUint32 {
		return metalConv2DConfig{}, tensor.ErrShapeMismatch
	}

	return metalConv2DConfig{
		batch:        uint32(inputDims[0]),
		inChannels:   uint32(inputDims[1]),
		inHeight:     uint32(inputDims[2]),
		inWidth:      uint32(inputDims[3]),
		outChannels:  uint32(weightDims[1]),
		kernelHeight: uint32(weightDims[2]),
		kernelWidth:  uint32(weightDims[3]),
		outHeight:    uint32(outDims[2]),
		outWidth:     uint32(outDims[3]),
	}, nil
}
