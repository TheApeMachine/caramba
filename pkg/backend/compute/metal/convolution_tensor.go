//go:build darwin && cgo

package metal

// #include "convolution.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type metalConv1DTensorConfig struct {
	input       computetensor.Float64Tensor
	weight      computetensor.Float64Tensor
	bias        computetensor.Float64Tensor
	outputShape computetensor.Shape
	batch       int
	inChannels  int
	length      int
	outChannels int
	kernelSize  int
	stride      int
	padding     int
	dilation    int
	groups      int
}

type metalConv2DTensorConfig struct {
	input          computetensor.Float64Tensor
	weight         computetensor.Float64Tensor
	bias           computetensor.Float64Tensor
	outputShape    computetensor.Shape
	batch          int
	inChannels     int
	height         int
	width          int
	outChannels    int
	kernelHeight   int
	kernelWidth    int
	strideHeight   int
	strideWidth    int
	padHeight      int
	padWidth       int
	dilationHeight int
	dilationWidth  int
	groups         int
}

func (m *ConvolutionOps) Conv1dTensor(
	input computetensor.Float64Tensor,
	weight computetensor.Float64Tensor,
	bias computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	batch int,
	inChannels int,
	length int,
	outChannels int,
	kernelSize int,
	stride int,
	padding int,
	dilation int,
	groups int,
) (computetensor.Float64Tensor, error) {
	return m.conv1dTensor(metalConv1DTensorConfig{
		input:       input,
		weight:      weight,
		bias:        bias,
		outputShape: outputShape,
		batch:       batch,
		inChannels:  inChannels,
		length:      length,
		outChannels: outChannels,
		kernelSize:  kernelSize,
		stride:      stride,
		padding:     padding,
		dilation:    dilation,
		groups:      groups,
	})
}

func (m *ConvolutionOps) conv1dTensor(
	config metalConv1DTensorConfig,
) (computetensor.Float64Tensor, error) {
	metalInput, metalWeight, metalBias, err := requireMetalConvolutionInputs(
		config.input, config.weight, config.bias,
	)
	if err != nil {
		return nil, err
	}

	outputDims := config.outputShape.Dims()
	if len(outputDims) != 3 {
		return nil, fmt.Errorf("metal.conv1d: output shape must be NCL rank 3")
	}

	if err := validateMetalConv1dLengths(
		metalInput.Len(),
		metalWeight.Len(),
		metalBias.Len(),
		config.batch,
		config.inChannels,
		config.length,
		config.outChannels,
		config.kernelSize,
		config.stride,
		config.padding,
		config.dilation,
		config.groups,
		outputDims[2],
	); err != nil {
		return nil, err
	}

	output, err := m.runtime.NewFloat32Tensor(config.outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	return m.dispatchConv1dTensor(
		metalInput, metalWeight, metalBias, output, config, outputDims[2],
	)
}

func (m *ConvolutionOps) dispatchConv1dTensor(
	metalInput *Tensor,
	metalWeight *Tensor,
	metalBias *Tensor,
	output *Tensor,
	config metalConv1DTensorConfig,
	lengthOut int,
) (computetensor.Float64Tensor, error) {
	rc := C.metal_conv1d_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(config.batch),
		C.int(config.inChannels),
		C.int(config.length),
		C.int(config.outChannels),
		C.int(config.kernelSize),
		C.int(config.stride),
		C.int(config.padding),
		C.int(config.dilation),
		C.int(config.groups),
		C.int(lengthOut),
		metalWeight.buffer,
		metalBias.buffer,
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_conv1d_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (m *ConvolutionOps) Conv2dTensor(
	input computetensor.Float64Tensor,
	weight computetensor.Float64Tensor,
	bias computetensor.Float64Tensor,
	outputShape computetensor.Shape,
	batch int,
	inChannels int,
	height int,
	width int,
	outChannels int,
	kernelHeight int,
	kernelWidth int,
	strideHeight int,
	strideWidth int,
	padHeight int,
	padWidth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
) (computetensor.Float64Tensor, error) {
	return m.conv2dTensor(metalConv2DTensorConfig{
		input:          input,
		weight:         weight,
		bias:           bias,
		outputShape:    outputShape,
		batch:          batch,
		inChannels:     inChannels,
		height:         height,
		width:          width,
		outChannels:    outChannels,
		kernelHeight:   kernelHeight,
		kernelWidth:    kernelWidth,
		strideHeight:   strideHeight,
		strideWidth:    strideWidth,
		padHeight:      padHeight,
		padWidth:       padWidth,
		dilationHeight: dilationHeight,
		dilationWidth:  dilationWidth,
		groups:         groups,
	})
}

func (m *ConvolutionOps) conv2dTensor(
	config metalConv2DTensorConfig,
) (computetensor.Float64Tensor, error) {
	metalInput, metalWeight, metalBias, err := requireMetalConvolutionInputs(
		config.input, config.weight, config.bias,
	)
	if err != nil {
		return nil, err
	}

	outputDims := config.outputShape.Dims()
	if len(outputDims) != 4 {
		return nil, fmt.Errorf("metal.conv2d: output shape must be NCHW rank 4")
	}

	if err := validateMetalConv2dLengths(
		metalInput.Len(),
		metalWeight.Len(),
		metalBias.Len(),
		config.batch,
		config.inChannels,
		config.height,
		config.width,
		config.outChannels,
		config.kernelHeight,
		config.kernelWidth,
		config.strideHeight,
		config.strideWidth,
		config.padHeight,
		config.padWidth,
		config.dilationHeight,
		config.dilationWidth,
		config.groups,
		outputDims[2],
		outputDims[3],
	); err != nil {
		return nil, err
	}

	output, err := m.runtime.NewFloat32Tensor(config.outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	return m.dispatchConv2dTensor(
		metalInput, metalWeight, metalBias, output, config, outputDims,
	)
}

func (m *ConvolutionOps) dispatchConv2dTensor(
	metalInput *Tensor,
	metalWeight *Tensor,
	metalBias *Tensor,
	output *Tensor,
	config metalConv2DTensorConfig,
	outputDims []int,
) (computetensor.Float64Tensor, error) {
	rc := C.metal_conv2d_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(config.batch),
		C.int(config.inChannels),
		C.int(config.height),
		C.int(config.width),
		C.int(config.outChannels),
		C.int(config.kernelHeight),
		C.int(config.kernelWidth),
		C.int(config.strideHeight),
		C.int(config.strideWidth),
		C.int(config.padHeight),
		C.int(config.padWidth),
		C.int(config.dilationHeight),
		C.int(config.dilationWidth),
		C.int(config.groups),
		C.int(outputDims[2]),
		C.int(outputDims[3]),
		metalWeight.buffer,
		metalBias.buffer,
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_conv2d_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func requireMetalConvolutionInputs(
	input computetensor.Float64Tensor,
	weight computetensor.Float64Tensor,
	bias computetensor.Float64Tensor,
) (*Tensor, *Tensor, *Tensor, error) {
	metalInput, err := requireMetalTensor(input)
	if err != nil {
		return nil, nil, nil, err
	}

	metalWeight, err := requireMetalTensor(weight)
	if err != nil {
		return nil, nil, nil, err
	}

	metalBias, err := requireMetalTensor(bias)
	if err != nil {
		return nil, nil, nil, err
	}

	return metalInput, metalWeight, metalBias, nil
}
