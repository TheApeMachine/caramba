//go:build darwin && cgo

package metal

// #include "pooling.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (poolingOps *PoolingOps) MaxPool2dTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	params MaxPool2dParams,
) (computetensor.Tensor, error) {
	metalInput, spec, err := requireMetalPool2dInput(input, outputShape)
	if err != nil {
		return nil, err
	}

	if err := validateMetalPool2dOutput(spec, params); err != nil {
		return nil, err
	}

	output, err := poolingOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_max_pool2d_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(spec.batch), C.int(spec.channels), C.int(spec.height), C.int(spec.width),
		C.int(params.KernelH), C.int(params.KernelW),
		C.int(params.StrideH), C.int(params.StrideW),
		C.int(params.PadH), C.int(params.PadW),
		C.int(params.DilationH), C.int(params.DilationW),
		C.int(spec.outputHeight), C.int(spec.outputWidth),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: max_pool2d launch failed")
	}

	return output, nil
}

func (poolingOps *PoolingOps) AvgPool2dTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	params AvgPool2dParams,
) (computetensor.Tensor, error) {
	metalInput, spec, err := requireMetalPool2dInput(input, outputShape)
	if err != nil {
		return nil, err
	}

	if err := validateMetalPool2dOutput(spec, maxParamsFromAvg(params)); err != nil {
		return nil, err
	}

	output, err := poolingOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	countPad := 0
	if params.CountIncludePad {
		countPad = 1
	}

	rc := C.metal_avg_pool2d_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(spec.batch), C.int(spec.channels), C.int(spec.height), C.int(spec.width),
		C.int(params.KernelH), C.int(params.KernelW),
		C.int(params.StrideH), C.int(params.StrideW),
		C.int(params.PadH), C.int(params.PadW),
		C.int(params.DilationH), C.int(params.DilationW),
		C.int(spec.outputHeight), C.int(spec.outputWidth),
		C.int(countPad), C.int(params.DivisorOverride),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: avg_pool2d launch failed")
	}

	return output, nil
}

func (poolingOps *PoolingOps) AdaptiveAvgPool2dTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	return poolingOps.adaptivePool2dTensor(input, outputShape, true)
}

func (poolingOps *PoolingOps) AdaptiveMaxPool2dTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
) (computetensor.Tensor, error) {
	return poolingOps.adaptivePool2dTensor(input, outputShape, false)
}

func (poolingOps *PoolingOps) adaptivePool2dTensor(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
	average bool,
) (computetensor.Tensor, error) {
	metalInput, spec, err := requireMetalPool2dInput(input, outputShape)
	if err != nil {
		return nil, err
	}

	output, err := poolingOps.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_adaptive_max_pool2d_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(spec.batch), C.int(spec.channels), C.int(spec.height), C.int(spec.width),
		C.int(spec.outputHeight), C.int(spec.outputWidth),
	)
	if average {
		rc = C.metal_adaptive_avg_pool2d_tensor(
			metalInput.buffer,
			output.buffer,
			C.int(spec.batch), C.int(spec.channels), C.int(spec.height), C.int(spec.width),
			C.int(spec.outputHeight), C.int(spec.outputWidth),
		)
	}
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: adaptive pool2d launch failed")
	}

	return output, nil
}

type metalPool2dSpec struct {
	batch        int
	channels     int
	height       int
	width        int
	outputHeight int
	outputWidth  int
}

func requireMetalPool2dInput(
	input computetensor.Tensor,
	outputShape computetensor.Shape,
) (*Tensor, metalPool2dSpec, error) {
	metalInput, err := requireMetalTensor(input)
	if err != nil {
		return nil, metalPool2dSpec{}, err
	}

	inputDims := metalInput.shape.Dims()
	outputDims := outputShape.Dims()
	if len(inputDims) != 4 || len(outputDims) != 4 {
		return nil, metalPool2dSpec{}, fmt.Errorf("metal tensor: pool2d requires NCHW rank-4 tensors")
	}

	spec := metalPool2dSpec{
		batch:        inputDims[0],
		channels:     inputDims[1],
		height:       inputDims[2],
		width:        inputDims[3],
		outputHeight: outputDims[2],
		outputWidth:  outputDims[3],
	}

	if spec.batch != outputDims[0] || spec.channels != outputDims[1] {
		return nil, metalPool2dSpec{}, fmt.Errorf("metal tensor: pool2d output N,C must match input")
	}

	if spec.batch <= 0 || spec.channels <= 0 || spec.height <= 0 || spec.width <= 0 ||
		spec.outputHeight <= 0 || spec.outputWidth <= 0 {
		return nil, metalPool2dSpec{}, fmt.Errorf("metal tensor: pool2d dimensions must be positive")
	}

	return metalInput, spec, nil
}

func validateMetalPool2dOutput(spec metalPool2dSpec, params MaxPool2dParams) error {
	if params.KernelH <= 0 || params.KernelW <= 0 ||
		params.StrideH <= 0 || params.StrideW <= 0 ||
		params.DilationH <= 0 || params.DilationW <= 0 ||
		params.PadH < 0 || params.PadW < 0 {
		return fmt.Errorf("metal tensor: pool2d parameters are invalid")
	}

	expectedHeight := metalOutSize(
		spec.height, params.KernelH, params.StrideH,
		params.PadH, params.DilationH, params.CeilMode,
	)
	expectedWidth := metalOutSize(
		spec.width, params.KernelW, params.StrideW,
		params.PadW, params.DilationW, params.CeilMode,
	)

	if spec.outputHeight == expectedHeight && spec.outputWidth == expectedWidth {
		return nil
	}

	return fmt.Errorf(
		"metal tensor: pool2d output H,W=%d,%d must match %d,%d",
		spec.outputHeight,
		spec.outputWidth,
		expectedHeight,
		expectedWidth,
	)
}

func maxParamsFromAvg(params AvgPool2dParams) MaxPool2dParams {
	return MaxPool2dParams{
		KernelH:   params.KernelH,
		KernelW:   params.KernelW,
		StrideH:   params.StrideH,
		StrideW:   params.StrideW,
		PadH:      params.PadH,
		PadW:      params.PadW,
		DilationH: params.DilationH,
		DilationW: params.DilationW,
		CeilMode:  params.CeilMode,
	}
}
