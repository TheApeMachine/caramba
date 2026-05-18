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
	"github.com/theapemachine/caramba/pkg/dtype"
)

type metalElementDType int

const (
	metalElementDTypeFloat32 metalElementDType = iota
	metalElementDTypeFloat16
	metalElementDTypeBFloat16
)

func metalElementDTypeFor(storageDType dtype.DType) (metalElementDType, error) {
	switch storageDType {
	case dtype.Float32:
		return metalElementDTypeFloat32, nil
	case dtype.Float16:
		return metalElementDTypeFloat16, nil
	case dtype.BFloat16:
		return metalElementDTypeBFloat16, nil
	}

	return 0, tensor.ErrDTypeMismatch
}

func runMetalBinaryElementwise(
	operation metalBinaryFloat32Operation,
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
) error {
	leftTensor, rightTensor, outTensor, err := requireBinaryElementwiseTensors(left, right, out)
	if err != nil {
		return err
	}

	elementDType, err := metalElementDTypeFor(leftTensor.dtype)
	if err != nil {
		return err
	}

	if leftTensor.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outTensor, leftTensor, rightTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_binary_elementwise(
		leftTensor.bridge.device,
		C.int(operation),
		C.int(elementDType),
		leftTensor.buffer,
		rightTensor.buffer,
		outTensor.buffer,
		C.uint32_t(leftTensor.shape.Len()),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal binary elementwise: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalUnaryElementwise(
	operation metalUnaryFloat32Operation,
	input tensor.Tensor,
	out tensor.Tensor,
) error {
	inputTensor, outTensor, err := requireUnaryElementwiseTensors(input, out)
	if err != nil {
		return err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return err
	}

	if inputTensor.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outTensor, inputTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_unary_elementwise(
		inputTensor.bridge.device,
		C.int(operation),
		C.int(elementDType),
		inputTensor.buffer,
		outTensor.buffer,
		C.uint32_t(inputTensor.shape.Len()),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal unary elementwise: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func requireBinaryElementwiseTensors(
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	leftTensor, err := requireMetalTensor(left)
	if err != nil {
		return nil, nil, nil, err
	}

	rightTensor, err := requireMetalTensor(right)
	if err != nil {
		return nil, nil, nil, err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return nil, nil, nil, err
	}

	if leftTensor.dtype != rightTensor.dtype || leftTensor.dtype != outTensor.dtype {
		return nil, nil, nil, tensor.ErrDTypeMismatch
	}

	if leftTensor.shape.Len() > math.MaxUint32 {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	if !leftTensor.shape.Equal(rightTensor.shape) || !leftTensor.shape.Equal(outTensor.shape) {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	if leftTensor.bridge != rightTensor.bridge || leftTensor.bridge != outTensor.bridge {
		return nil, nil, nil, errors.New("metal binary elementwise: tensors belong to different Metal backends")
	}

	return leftTensor, rightTensor, outTensor, nil
}

func requireUnaryElementwiseTensors(
	input tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
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

	if inputTensor.shape.Len() > math.MaxUint32 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	if !inputTensor.shape.Equal(outTensor.shape) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	if inputTensor.bridge != outTensor.bridge {
		return nil, nil, errors.New("metal unary elementwise: tensors belong to different Metal backends")
	}

	return inputTensor, outTensor, nil
}
