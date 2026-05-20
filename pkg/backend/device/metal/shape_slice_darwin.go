//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func runMetalSlice(
	input tensor.Tensor,
	dim tensor.Tensor,
	start tensor.Tensor,
	end tensor.Tensor,
	out tensor.Tensor,
) error {
	inputTensor, outTensor, err := requireMetalSlice(input, dim, start, end, out)
	if err != nil {
		return err
	}

	if outTensor.bytes == 0 {
		return nil
	}

	sliceParams, err := computeMetalSliceDispatchParams(inputTensor, dim, start, end, outTensor)
	if err != nil {
		return err
	}

	elementDType, err := metalShapeElementDType(inputTensor)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(outTensor, inputTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_slice_bytes(
		inputTensor.bridge.device,
		C.int(elementDType),
		inputTensor.buffer,
		outTensor.buffer,
		C.uint32_t(sliceParams.sliceLen),
		C.uint32_t(sliceParams.inputDimSize),
		C.uint32_t(sliceParams.innerBytes),
		C.uint32_t(sliceParams.start),
		C.uint32_t(outTensor.bytes),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal slice: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

type metalSliceDispatchParams struct {
	sliceLen     int
	inputDimSize int
	innerBytes   int
	start        int
}

func computeMetalSliceDispatchParams(
	inputTensor *metalTensor,
	dim tensor.Tensor,
	start tensor.Tensor,
	end tensor.Tensor,
	outTensor *metalTensor,
) (metalSliceDispatchParams, error) {
	dimValue, err := metalInt32Scalar(dim, inputTensor.bridge)
	if err != nil {
		return metalSliceDispatchParams{}, err
	}

	startValue, err := metalInt32Scalar(start, inputTensor.bridge)
	if err != nil {
		return metalSliceDispatchParams{}, err
	}

	endValue, err := metalInt32Scalar(end, inputTensor.bridge)
	if err != nil {
		return metalSliceDispatchParams{}, err
	}

	inDims := inputTensor.shape.Dims()
	outDims := outTensor.shape.Dims()
	rank := len(inDims)

	if int(dimValue) < 0 || int(dimValue) >= rank {
		return metalSliceDispatchParams{}, tensor.ErrShapeMismatch
	}

	dimSize := inDims[dimValue]
	sliceEnd := int(endValue)

	if sliceEnd == 0 {
		sliceEnd = dimSize
	}

	if int(startValue) < 0 || sliceEnd < int(startValue) || sliceEnd > dimSize {
		return metalSliceDispatchParams{}, tensor.ErrShapeMismatch
	}

	sliceLen := sliceEnd - int(startValue)

	for axis := 0; axis < rank; axis++ {
		if axis == int(dimValue) {
			if outDims[axis] != sliceLen {
				return metalSliceDispatchParams{}, tensor.ErrShapeMismatch
			}

			continue
		}

		if inDims[axis] != outDims[axis] {
			return metalSliceDispatchParams{}, tensor.ErrShapeMismatch
		}
	}

	elementBytes, err := inputTensor.dtype.Size()
	if err != nil {
		return metalSliceDispatchParams{}, err
	}

	inner := 1
	for axis := int(dimValue) + 1; axis < rank; axis++ {
		inner *= inDims[axis]
	}

	innerBytes := inner * elementBytes

	if err := requireUint32(sliceLen); err != nil {
		return metalSliceDispatchParams{}, err
	}

	if err := requireUint32(dimSize); err != nil {
		return metalSliceDispatchParams{}, err
	}

	if err := requireUint32(innerBytes); err != nil {
		return metalSliceDispatchParams{}, err
	}

	if err := requireUint32(int(startValue)); err != nil {
		return metalSliceDispatchParams{}, err
	}

	return metalSliceDispatchParams{
		sliceLen:     sliceLen,
		inputDimSize: dimSize,
		innerBytes:   innerBytes,
		start:        int(startValue),
	}, nil
}
