//go:build darwin && cgo

package metal

// #include "activation.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

/*
ReLUTensor launches a Metal ReLU kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) ReLUTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "relu", 0)
}

/*
LeakyReLUTensor launches a Metal leaky ReLU kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) LeakyReLUTensor(
	input computetensor.Float64Tensor, alpha float64,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "leaky_relu", alpha)
}

/*
GELUTensor launches a Metal GELU kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) GELUTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "gelu", 0)
}

/*
TanhTensor launches a Metal tanh kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) TanhTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "tanh", 0)
}

/*
SigmoidTensor launches a Metal sigmoid kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) SigmoidTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "sigmoid", 0)
}

/*
SwishTensor launches a Metal Swish kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) SwishTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "swish", 0)
}

/*
SELUTensor launches a Metal SELU kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) SELUTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return metalActivation.unaryTensor(input, "selu", 0)
}

/*
SwiGLUTensor launches a Metal SwiGLU kernel directly against resident buffers.
*/
func (metalActivation *MetalActivation) SwiGLUTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	outputShape, err := metalSwiGLUOutputShape(metalInput.shape)

	if err != nil {
		return nil, err
	}

	output, err := metalActivation.runtime.NewFloat32Tensor(
		outputShape,
		MetalAllocationTensor,
	)

	if err != nil {
		return nil, err
	}

	if outputShape.Len() == 0 {
		return output, nil
	}

	inputDims := metalInput.shape.Dims()
	inputWidth := metalInput.Len()

	if len(inputDims) > 0 {
		inputWidth = inputDims[len(inputDims)-1]
	}

	rc := C.metal_swiglu_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(outputShape.Len()),
		C.int(inputWidth),
	)

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: swiglu launch failed")
	}

	return output, nil
}

func (metalActivation *MetalActivation) unaryTensor(
	input computetensor.Float64Tensor, name string, alpha float64,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)

	if err != nil {
		return nil, err
	}

	output, err := metalActivation.runtime.NewFloat32Tensor(
		metalInput.shape,
		MetalAllocationTensor,
	)

	if err != nil {
		return nil, err
	}

	var rc C.int

	switch name {
	case "relu":
		rc = C.metal_relu_tensor(metalInput.buffer, output.buffer, C.int(metalInput.Len()))
	case "leaky_relu":
		rc = C.metal_leaky_relu_tensor(
			metalInput.buffer, output.buffer, C.float(alpha), C.int(metalInput.Len()),
		)
	case "gelu":
		rc = C.metal_gelu_tensor(metalInput.buffer, output.buffer, C.int(metalInput.Len()))
	case "tanh":
		rc = C.metal_tanh_tensor(metalInput.buffer, output.buffer, C.int(metalInput.Len()))
	case "sigmoid":
		rc = C.metal_sigmoid_tensor(metalInput.buffer, output.buffer, C.int(metalInput.Len()))
	case "swish":
		rc = C.metal_swish_tensor(metalInput.buffer, output.buffer, C.int(metalInput.Len()))
	case "selu":
		rc = C.metal_selu_tensor(metalInput.buffer, output.buffer, C.int(metalInput.Len()))
	default:
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: unknown unary kernel %q", name)
	}

	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: %s launch failed", name)
	}

	return output, nil
}

func metalSwiGLUOutputShape(shape computetensor.Shape) (computetensor.Shape, error) {
	dimsCopy := append([]int(nil), shape.Dims()...)

	if shape.Len()%2 != 0 {
		return computetensor.Shape{}, fmt.Errorf("metal tensor: swiglu input length must be even")
	}

	if len(dimsCopy) == 0 {
		return computetensor.NewShape([]int{shape.Len() / 2})
	}

	lastIndex := len(dimsCopy) - 1

	if dimsCopy[lastIndex]%2 != 0 {
		return computetensor.Shape{}, fmt.Errorf("metal tensor: swiglu final dimension must be even")
	}

	dimsCopy[lastIndex] /= 2

	return computetensor.NewShape(dimsCopy)
}
