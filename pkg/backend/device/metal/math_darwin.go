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

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type metalInvSqrtDimScaleConfig struct {
	input        *metalTensor
	dim          *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
}

type metalLogSumExpConfig struct {
	input        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	rows         uint32
	cols         uint32
}

type metalOuterConfig struct {
	left         *metalTensor
	right        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	rows         uint32
	cols         uint32
}

func runMetalInvSqrtDimScale(input tensor.Tensor, dim tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalInvSqrtDimScale(input, dim, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.dim)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_inv_sqrt_dim_scale(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.dim.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalMathDispatch("inv_sqrt_dim_scale", token, rc, status)
}

func runMetalLogSumExp(input tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalLogSumExp(input, out)
	if err != nil {
		return err
	}

	if config.rows == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_logsumexp(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	return finishMetalMathDispatch("logsumexp", token, rc, status)
}

func runMetalOuter(left tensor.Tensor, right tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalOuter(left, right, out)
	if err != nil {
		return err
	}

	if config.rows == 0 || config.cols == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.left, config.right)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_outer(
		config.left.bridge.device,
		C.int(config.elementDType),
		config.left.buffer,
		config.right.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	return finishMetalMathDispatch("outer", token, rc, status)
}

func requireMetalInvSqrtDimScale(
	input tensor.Tensor,
	dim tensor.Tensor,
	out tensor.Tensor,
) (metalInvSqrtDimScaleConfig, error) {
	tensors, err := requireMetalTensors(input, dim, out)
	if err != nil {
		return metalInvSqrtDimScaleConfig{}, err
	}

	inputTensor, dimTensor, outTensor := tensors[0], tensors[1], tensors[2]
	if dimTensor.dtype != dtype.Int32 || inputTensor.dtype != outTensor.dtype {
		return metalInvSqrtDimScaleConfig{}, tensor.ErrDTypeMismatch
	}

	if inputTensor.bridge != dimTensor.bridge || inputTensor.bridge != outTensor.bridge {
		return metalInvSqrtDimScaleConfig{}, errors.New("metal math: tensors belong to different Metal backends")
	}

	if inputTensor.shape.Len() != outTensor.shape.Len() || dimTensor.shape.Len() < 1 {
		return metalInvSqrtDimScaleConfig{}, tensor.ErrShapeMismatch
	}

	if inputTensor.shape.Len() > math.MaxUint32 {
		return metalInvSqrtDimScaleConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalInvSqrtDimScaleConfig{}, err
	}

	return metalInvSqrtDimScaleConfig{
		input: inputTensor, dim: dimTensor, out: outTensor,
		elementDType: elementDType, count: uint32(inputTensor.shape.Len()),
	}, nil
}

func requireMetalLogSumExp(input tensor.Tensor, out tensor.Tensor) (metalLogSumExpConfig, error) {
	inputTensor, outTensor, err := requireMetalMathSameDType(input, out)
	if err != nil {
		return metalLogSumExpConfig{}, err
	}

	rows, cols, err := metalLogSumExpDims(inputTensor, outTensor)
	if err != nil {
		return metalLogSumExpConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalLogSumExpConfig{}, err
	}

	return metalLogSumExpConfig{
		input: inputTensor, out: outTensor, elementDType: elementDType,
		rows: uint32(rows), cols: uint32(cols),
	}, nil
}

func requireMetalOuter(
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
) (metalOuterConfig, error) {
	tensors, err := requireMetalTensors(left, right, out)
	if err != nil {
		return metalOuterConfig{}, err
	}

	leftTensor, rightTensor, outTensor := tensors[0], tensors[1], tensors[2]
	if leftTensor.dtype != rightTensor.dtype || leftTensor.dtype != outTensor.dtype {
		return metalOuterConfig{}, tensor.ErrDTypeMismatch
	}

	if leftTensor.bridge != rightTensor.bridge || leftTensor.bridge != outTensor.bridge {
		return metalOuterConfig{}, errors.New("metal math: tensors belong to different Metal backends")
	}

	rows := leftTensor.shape.Len()
	cols := rightTensor.shape.Len()
	if rows > math.MaxUint32 || cols > math.MaxUint32 || rows*cols != outTensor.shape.Len() {
		return metalOuterConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(leftTensor.dtype)
	if err != nil {
		return metalOuterConfig{}, err
	}

	return metalOuterConfig{
		left: leftTensor, right: rightTensor, out: outTensor,
		elementDType: elementDType, rows: uint32(rows), cols: uint32(cols),
	}, nil
}

func requireMetalMathSameDType(first tensor.Tensor, second tensor.Tensor) (*metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(first, second)
	if err != nil {
		return nil, nil, err
	}

	if tensors[0].dtype != tensors[1].dtype {
		return nil, nil, tensor.ErrDTypeMismatch
	}

	if tensors[0].bridge != tensors[1].bridge {
		return nil, nil, errors.New("metal math: tensors belong to different Metal backends")
	}

	return tensors[0], tensors[1], nil
}

func metalLogSumExpDims(input *metalTensor, out *metalTensor) (int, int, error) {
	dims := input.shape.Dims()
	if len(dims) == 0 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	cols := dims[len(dims)-1]
	if cols == 0 || input.shape.Len()%cols != 0 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	rows := input.shape.Len() / cols
	if out.shape.Len() != rows || rows > math.MaxUint32 || cols > math.MaxUint32 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	return rows, cols, nil
}

func finishMetalMathDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
