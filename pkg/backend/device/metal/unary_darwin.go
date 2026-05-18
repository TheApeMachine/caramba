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

type metalUnaryFloat32Operation int

const (
	metalUnaryFloat32Relu metalUnaryFloat32Operation = iota
	metalUnaryFloat32Abs
	metalUnaryFloat32Neg
	metalUnaryFloat32Square
	metalUnaryFloat32Recip
	metalUnaryFloat32Sqrt
	metalUnaryFloat32Sign
)

func runMetalUnaryFloat32(
	operation metalUnaryFloat32Operation,
	input tensor.Tensor,
	out tensor.Tensor,
) error {
	inputTensor, err := requireMetalTensor(input)
	if err != nil {
		return err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return err
	}

	if inputTensor.dtype != dtype.Float32 || outTensor.dtype != dtype.Float32 {
		return tensor.ErrDTypeMismatch
	}

	if !inputTensor.shape.Equal(outTensor.shape) {
		return tensor.ErrShapeMismatch
	}

	if inputTensor.bridge != outTensor.bridge {
		return errors.New("metal unary float32: tensors belong to different Metal backends")
	}

	if inputTensor.shape.Len() > math.MaxUint32 {
		return tensor.ErrShapeMismatch
	}

	if inputTensor.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outTensor, inputTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_unary_float32(
		inputTensor.bridge.device,
		C.int(operation),
		inputTensor.buffer,
		outTensor.buffer,
		C.uint32_t(inputTensor.shape.Len()),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal unary float32: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}
