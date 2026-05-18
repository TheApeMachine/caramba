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

const maxMetalUint32 = int64(1<<32 - 1)

func runMetalReshape(input tensor.Tensor, out tensor.Tensor) error {
	inputTensor, outTensor, err := requireMetalShapeCopy(input, out)
	if err != nil {
		return err
	}

	return dispatchMetalCopyBytes(inputTensor, outTensor)
}

func runMetalMergeHeads(input tensor.Tensor, out tensor.Tensor) error {
	inputTensor, outTensor, err := requireMetalShapeCopy(input, out)
	if err != nil {
		return err
	}

	if err := requireMergeHeadsShape(inputTensor, outTensor); err != nil {
		return err
	}

	return dispatchMetalCopyBytes(inputTensor, outTensor)
}

func runMetalSplitHeads(input tensor.Tensor, out tensor.Tensor) error {
	inputTensor, outTensor, err := requireMetalShapeCopy(input, out)
	if err != nil {
		return err
	}

	if err := requireSplitHeadsShape(inputTensor, outTensor); err != nil {
		return err
	}

	return dispatchMetalCopyBytes(inputTensor, outTensor)
}

func runMetalViewAsHeads(input tensor.Tensor, heads tensor.Tensor, out tensor.Tensor) error {
	inputTensor, outTensor, err := requireMetalShapeCopy(input, out)
	if err != nil {
		return err
	}

	headCount, err := metalInt32Scalar(heads, inputTensor.bridge)
	if err != nil {
		return err
	}

	if err := requireViewAsHeadsShape(inputTensor, outTensor, headCount); err != nil {
		return err
	}

	return dispatchMetalCopyBytes(inputTensor, outTensor)
}

func runMetalConcat(left tensor.Tensor, right tensor.Tensor, out tensor.Tensor) error {
	leftTensor, rightTensor, outTensor, err := requireMetalConcat(left, right, out)
	if err != nil {
		return err
	}

	if outTensor.bytes == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outTensor, leftTensor, rightTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_concat_bytes(
		leftTensor.bridge.device,
		leftTensor.buffer,
		rightTensor.buffer,
		outTensor.buffer,
		C.uint32_t(leftTensor.bytes),
		C.uint32_t(rightTensor.bytes),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal concat: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalSplit2(input tensor.Tensor, left tensor.Tensor, right tensor.Tensor) error {
	inputTensor, leftTensor, rightTensor, err := requireMetalSplit2(input, left, right)
	if err != nil {
		return err
	}

	if inputTensor.bytes == 0 {
		return nil
	}

	token, err := metalCompletions.BeginMany([]*metalTensor{leftTensor, rightTensor}, inputTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_split2_bytes(
		inputTensor.bridge.device,
		inputTensor.buffer,
		leftTensor.buffer,
		rightTensor.buffer,
		C.uint32_t(leftTensor.bytes),
		C.uint32_t(rightTensor.bytes),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal split2: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalLastToken(input tensor.Tensor, out tensor.Tensor) error {
	inputTensor, outTensor, err := requireMetalLastToken(input, out)
	if err != nil {
		return err
	}

	if outTensor.bytes == 0 {
		return nil
	}

	dims := inputTensor.shape.Dims()
	elementBytes, _ := inputTensor.dtype.Size()
	hiddenBytes := dims[2] * elementBytes

	token, err := metalCompletions.Begin(outTensor, inputTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_last_token_bytes(
		inputTensor.bridge.device,
		inputTensor.buffer,
		outTensor.buffer,
		C.uint32_t(dims[1]),
		C.uint32_t(hiddenBytes),
		C.uint32_t(outTensor.bytes),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal last_token: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalTranspose2D(input tensor.Tensor, out tensor.Tensor) error {
	inputTensor, outTensor, err := requireMetalTranspose2D(input, out)
	if err != nil {
		return err
	}

	if outTensor.bytes == 0 {
		return nil
	}

	dims := inputTensor.shape.Dims()
	elementBytes, _ := inputTensor.dtype.Size()
	token, err := metalCompletions.Begin(outTensor, inputTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_transpose2d_bytes(
		inputTensor.bridge.device,
		inputTensor.buffer,
		outTensor.buffer,
		C.uint32_t(dims[0]),
		C.uint32_t(dims[1]),
		C.uint32_t(elementBytes),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal transpose2d: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalUpsampleNearest2D(input tensor.Tensor, out tensor.Tensor) error {
	inputTensor, outTensor, err := requireMetalUpsampleNearest2D(input, out)
	if err != nil {
		return err
	}

	if outTensor.bytes == 0 {
		return nil
	}

	inDims := inputTensor.shape.Dims()
	outDims := outTensor.shape.Dims()
	elementBytes, _ := inputTensor.dtype.Size()
	token, err := metalCompletions.Begin(outTensor, inputTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_upsample_nearest2d_bytes(
		inputTensor.bridge.device,
		inputTensor.buffer,
		outTensor.buffer,
		C.uint32_t(inDims[1]),
		C.uint32_t(inDims[2]),
		C.uint32_t(inDims[3]),
		C.uint32_t(outDims[2]),
		C.uint32_t(outDims[3]),
		C.uint32_t(elementBytes),
		C.uint32_t(outTensor.shape.Len()),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal upsample_nearest2d: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func dispatchMetalCopyBytes(inputTensor *metalTensor, outTensor *metalTensor) error {
	if outTensor.bytes == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(outTensor, inputTensor)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_copy_bytes(
		inputTensor.bridge.device,
		inputTensor.buffer,
		outTensor.buffer,
		C.uint32_t(outTensor.bytes),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal copy bytes: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}
