//go:build darwin && cgo

package metal

// #include "metal_kernel_math.h"
import "C"

import (
	"fmt"

	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func (m *MathOps) InvSqrtDimScaleTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)
	if err != nil {
		return nil, err
	}

	dimensions := metalInput.shape.Dims()
	if len(dimensions) == 0 {
		return nil, fmt.Errorf("metal tensor: inv_sqrt_dim_scale input shape is required")
	}

	return m.dispatchScaleTensor(metalInput, dimensions[len(dimensions)-1])
}

func (m *MathOps) ExpTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return m.unaryTensor(input, "exp")
}

func (m *MathOps) SinTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return m.unaryTensor(input, "sin")
}

func (m *MathOps) CosTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return m.unaryTensor(input, "cos")
}

func (m *MathOps) LogTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return m.unaryTensor(input, "log")
}

func (m *MathOps) SignTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	return m.unaryTensor(input, "sign")
}

func (m *MathOps) DropoutTensor(
	input computetensor.Float64Tensor,
	probability float64,
	training bool,
	seed int,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)
	if err != nil {
		return nil, err
	}

	if probability < 0 || probability >= 1 {
		return nil, fmt.Errorf("metal tensor: dropout probability must be in [0,1)")
	}

	output, err := m.runtime.NewFloat32Tensor(metalInput.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	trainingValue := 0
	if training {
		trainingValue = 1
	}

	rc := C.metal_dropout_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(metalInput.Len()),
		C.float(probability),
		C.int(trainingValue),
		C.int(seed),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: dropout launch failed")
	}

	return output, nil
}

func (m *MathOps) OuterTensor(
	left computetensor.Float64Tensor,
	right computetensor.Float64Tensor,
	outputShape computetensor.Shape,
) (computetensor.Float64Tensor, error) {
	metalLeft, metalRight, rows, columns, err := requireMetalOuterInputs(
		left, right, outputShape,
	)
	if err != nil {
		return nil, err
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_outer_tensor(
		metalLeft.buffer,
		metalRight.buffer,
		output.buffer,
		C.int(rows),
		C.int(columns),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: outer launch failed")
	}

	return output, nil
}

func (m *MathOps) SoftmaxTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	metalInput, numRows, dimSize, err := requireMetalRowReduction(input, "softmax")
	if err != nil {
		return nil, err
	}

	output, err := m.runtime.NewFloat32Tensor(metalInput.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_softmax_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(numRows),
		C.int(dimSize),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: softmax launch failed")
	}

	return output, nil
}

func (m *MathOps) LogSumExpTensor(
	input computetensor.Float64Tensor,
) (computetensor.Float64Tensor, error) {
	metalInput, numRows, dimSize, err := requireMetalRowReduction(input, "logsumexp")
	if err != nil {
		return nil, err
	}

	outputShape, err := metalLogSumExpOutputShape(metalInput.shape)
	if err != nil {
		return nil, err
	}

	output, err := m.runtime.NewFloat32Tensor(outputShape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_logsumexp_tensor(
		metalInput.buffer,
		output.buffer,
		C.int(numRows),
		C.int(dimSize),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: logsumexp launch failed")
	}

	return output, nil
}

func (m *MathOps) unaryTensor(
	input computetensor.Float64Tensor,
	name string,
) (computetensor.Float64Tensor, error) {
	metalInput, err := requireMetalTensor(input)
	if err != nil {
		return nil, err
	}

	output, err := m.runtime.NewFloat32Tensor(metalInput.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := m.dispatchUnaryTensor(metalInput, output, name)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: %s launch failed", name)
	}

	return output, nil
}

func (m *MathOps) dispatchUnaryTensor(
	input *Tensor,
	output *Tensor,
	name string,
) C.int {
	switch name {
	case "exp":
		return C.metal_exp_tensor(input.buffer, output.buffer, C.int(input.Len()))
	case "sin":
		return C.metal_sin_tensor(input.buffer, output.buffer, C.int(input.Len()))
	case "cos":
		return C.metal_cos_tensor(input.buffer, output.buffer, C.int(input.Len()))
	case "log":
		return C.metal_log_tensor(input.buffer, output.buffer, C.int(input.Len()))
	case "sign":
		return C.metal_sign_tensor(input.buffer, output.buffer, C.int(input.Len()))
	default:
		return -1
	}
}

func (m *MathOps) dispatchScaleTensor(
	input *Tensor,
	dimSize int,
) (computetensor.Float64Tensor, error) {
	if dimSize <= 0 {
		return nil, fmt.Errorf("metal tensor: inv_sqrt_dim_scale final dimension must be positive")
	}

	output, err := m.runtime.NewFloat32Tensor(input.shape, MetalAllocationTensor)
	if err != nil {
		return nil, err
	}

	rc := C.metal_inv_sqrt_dim_scale_tensor(
		input.buffer,
		output.buffer,
		C.int(input.Len()),
		C.int(dimSize),
	)
	if rc != 0 {
		_ = output.Close()

		return nil, fmt.Errorf("metal tensor: inv_sqrt_dim_scale launch failed")
	}

	return output, nil
}

func requireMetalRowReduction(
	input computetensor.Float64Tensor,
	operation string,
) (*Tensor, int, int, error) {
	metalInput, err := requireMetalTensor(input)
	if err != nil {
		return nil, 0, 0, err
	}

	dimensions := metalInput.shape.Dims()
	if len(dimensions) == 0 {
		return nil, 0, 0, fmt.Errorf("metal tensor: %s input shape is required", operation)
	}

	dimSize := dimensions[len(dimensions)-1]
	if dimSize <= 0 {
		return nil, 0, 0, fmt.Errorf(
			"metal tensor: %s final dimension must be positive",
			operation,
		)
	}

	if metalInput.Len() == 0 || metalInput.Len()%dimSize != 0 {
		return nil, 0, 0, fmt.Errorf(
			"metal tensor: %s input length %d must divide final dimension %d",
			operation,
			metalInput.Len(),
			dimSize,
		)
	}

	return metalInput, metalInput.Len() / dimSize, dimSize, nil
}

func metalLogSumExpOutputShape(
	inputShape computetensor.Shape,
) (computetensor.Shape, error) {
	dimensions := inputShape.Dims()
	outputDims := []int{1}

	if len(dimensions) > 1 {
		outputDims = append([]int(nil), dimensions[:len(dimensions)-1]...)
	}

	return computetensor.NewShape(outputDims)
}

func requireMetalOuterInputs(
	left computetensor.Float64Tensor,
	right computetensor.Float64Tensor,
	outputShape computetensor.Shape,
) (*Tensor, *Tensor, int, int, error) {
	metalLeft, err := requireMetalTensor(left)
	if err != nil {
		return nil, nil, 0, 0, err
	}

	metalRight, err := requireMetalTensor(right)
	if err != nil {
		return nil, nil, 0, 0, err
	}

	outputDims := outputShape.Dims()
	if len(outputDims) < 2 {
		return nil, nil, 0, 0, fmt.Errorf("metal tensor: outer output shape must be rank >= 2")
	}

	rows := outputDims[0]
	columns := outputDims[1]
	if rows <= 0 || columns <= 0 {
		return nil, nil, 0, 0, fmt.Errorf("metal tensor: outer dimensions must be positive")
	}

	if metalLeft.Len() != rows || metalRight.Len() != columns {
		return nil, nil, 0, 0, fmt.Errorf(
			"metal tensor: outer input lengths %d,%d must match output shape %d,%d",
			metalLeft.Len(),
			metalRight.Len(),
			rows,
			columns,
		)
	}

	if outputShape.Len() != rows*columns {
		return nil, nil, 0, 0, fmt.Errorf("metal tensor: outer output shape must equal rows*columns")
	}

	return metalLeft, metalRight, rows, columns, nil
}
