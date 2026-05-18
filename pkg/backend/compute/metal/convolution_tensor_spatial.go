//go:build darwin && cgo

package metal

// #include "convolution.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type metalConv3DTensorConfig struct {
	input          computetensor.Tensor
	weight         computetensor.Tensor
	bias           computetensor.Tensor
	outputShape    computetensor.Shape
	batch          int
	inChannels     int
	depth          int
	height         int
	width          int
	outChannels    int
	kernelDepth    int
	kernelHeight   int
	kernelWidth    int
	strideDepth    int
	strideHeight   int
	strideWidth    int
	padDepth       int
	padHeight      int
	padWidth       int
	dilationDepth  int
	dilationHeight int
	dilationWidth  int
	groups         int
}

type metalConvTranspose2DTensorConfig struct {
	input          computetensor.Tensor
	weight         computetensor.Tensor
	bias           computetensor.Tensor
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
	outPadHeight   int
	outPadWidth    int
}

func (m *ConvolutionOps) Conv3dTensor(
	input computetensor.Tensor,
	weight computetensor.Tensor,
	bias computetensor.Tensor,
	outputShape computetensor.Shape,
	batch int,
	inChannels int,
	depth int,
	height int,
	width int,
	outChannels int,
	kernelDepth int,
	kernelHeight int,
	kernelWidth int,
	strideDepth int,
	strideHeight int,
	strideWidth int,
	padDepth int,
	padHeight int,
	padWidth int,
	dilationDepth int,
	dilationHeight int,
	dilationWidth int,
	groups int,
) (computetensor.Tensor, error) {
	return m.conv3dTensor(metalConv3DTensorConfig{
		input:          input,
		weight:         weight,
		bias:           bias,
		outputShape:    outputShape,
		batch:          batch,
		inChannels:     inChannels,
		depth:          depth,
		height:         height,
		width:          width,
		outChannels:    outChannels,
		kernelDepth:    kernelDepth,
		kernelHeight:   kernelHeight,
		kernelWidth:    kernelWidth,
		strideDepth:    strideDepth,
		strideHeight:   strideHeight,
		strideWidth:    strideWidth,
		padDepth:       padDepth,
		padHeight:      padHeight,
		padWidth:       padWidth,
		dilationDepth:  dilationDepth,
		dilationHeight: dilationHeight,
		dilationWidth:  dilationWidth,
		groups:         groups,
	})
}

func (m *ConvolutionOps) conv3dTensor(
	config metalConv3DTensorConfig,
) (computetensor.Tensor, error) {
	metalInput, metalWeight, metalBias, err := requireMetalConvolutionInputs(
		config.input, config.weight, config.bias,
	)
	if err != nil {
		return nil, err
	}

	outputDims := config.outputShape.Dims()
	if len(outputDims) != 5 {
		return nil, fmt.Errorf("metal.conv3d: output shape must be NCDHW rank 5")
	}

	if err := validateMetalConv3dLengths(
		metalInput.Len(),
		metalWeight.Len(),
		metalBias.Len(),
		config.batch,
		config.inChannels,
		config.depth,
		config.height,
		config.width,
		config.outChannels,
		config.kernelDepth,
		config.kernelHeight,
		config.kernelWidth,
		config.strideDepth,
		config.strideHeight,
		config.strideWidth,
		config.padDepth,
		config.padHeight,
		config.padWidth,
		config.dilationDepth,
		config.dilationHeight,
		config.dilationWidth,
		config.groups,
		outputDims[2],
		outputDims[3],
		outputDims[4],
	); err != nil {
		return nil, err
	}

	output, err := m.runtime.NewFloat32Tensor(config.outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	return m.dispatchConv3dTensor(
		metalInput, metalWeight, metalBias, output, config, outputDims,
	)
}

func (m *ConvolutionOps) dispatchConv3dTensor(
	metalInput *Tensor,
	metalWeight *Tensor,
	metalBias *Tensor,
	output *Tensor,
	config metalConv3DTensorConfig,
	outputDims []int,
) (computetensor.Tensor, error) {
	rc := C.metal_conv3d_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(config.batch),
		C.int(config.inChannels),
		C.int(config.depth),
		C.int(config.height),
		C.int(config.width),
		C.int(config.outChannels),
		C.int(config.kernelDepth),
		C.int(config.kernelHeight),
		C.int(config.kernelWidth),
		C.int(config.strideDepth),
		C.int(config.strideHeight),
		C.int(config.strideWidth),
		C.int(config.padDepth),
		C.int(config.padHeight),
		C.int(config.padWidth),
		C.int(config.dilationDepth),
		C.int(config.dilationHeight),
		C.int(config.dilationWidth),
		C.int(config.groups),
		C.int(outputDims[2]),
		C.int(outputDims[3]),
		C.int(outputDims[4]),
		metalWeight.buffer,
		metalBias.buffer,
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal_conv3d_tensor failed (rc=%d)", rc)
	}

	return output, nil
}

func (m *ConvolutionOps) ConvTranspose2dTensor(
	input computetensor.Tensor,
	weight computetensor.Tensor,
	bias computetensor.Tensor,
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
	outPadHeight int,
	outPadWidth int,
) (computetensor.Tensor, error) {
	return m.convTranspose2dTensor(metalConvTranspose2DTensorConfig{
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
		outPadHeight:   outPadHeight,
		outPadWidth:    outPadWidth,
	})
}

func (m *ConvolutionOps) convTranspose2dTensor(
	config metalConvTranspose2DTensorConfig,
) (computetensor.Tensor, error) {
	metalInput, metalWeight, metalBias, err := requireMetalConvolutionInputs(
		config.input, config.weight, config.bias,
	)
	if err != nil {
		return nil, err
	}

	outputDims := config.outputShape.Dims()
	if len(outputDims) != 4 {
		return nil, fmt.Errorf("metal.conv_transpose2d: output shape must be NCHW rank 4")
	}

	if err := validateMetalConvTranspose2dLengths(
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
		config.outPadHeight,
		config.outPadWidth,
		outputDims[2],
		outputDims[3],
	); err != nil {
		return nil, err
	}

	output, err := m.runtime.NewFloat32Tensor(config.outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	return m.dispatchConvTranspose2dTensor(
		metalInput, metalWeight, metalBias, output, config, outputDims,
	)
}

func (m *ConvolutionOps) dispatchConvTranspose2dTensor(
	metalInput *Tensor,
	metalWeight *Tensor,
	metalBias *Tensor,
	output *Tensor,
	config metalConvTranspose2DTensorConfig,
	outputDims []int,
) (computetensor.Tensor, error) {
	rc := C.metal_conv_transpose2d_tensor(
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

		return nil, fmt.Errorf("metal_conv_transpose2d_tensor failed (rc=%d)", rc)
	}

	return output, nil
}
