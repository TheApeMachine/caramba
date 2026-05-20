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
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

type metalSoftmaxConfig struct {
	input        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	rows         uint32
	cols         uint32
}

func runMetalSoftmax(input tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalSoftmax(input, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_softmax(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal softmax: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func requireMetalSoftmax(
	input tensor.Tensor,
	out tensor.Tensor,
) (metalSoftmaxConfig, error) {
	inputTensor, outTensor, err := requireUnaryElementwiseTensors(input, out)
	if err != nil {
		return metalSoftmaxConfig{}, err
	}

	rows, cols, err := metalSoftmaxDims(inputTensor)
	if err != nil {
		return metalSoftmaxConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalSoftmaxConfig{}, err
	}

	return metalSoftmaxConfig{
		input:        inputTensor,
		out:          outTensor,
		elementDType: elementDType,
		rows:         uint32(rows),
		cols:         uint32(cols),
	}, nil
}

func metalSoftmaxDims(input *metalTensor) (int, int, error) {
	dims := input.shape.Dims()

	if len(dims) == 0 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	cols := dims[len(dims)-1]
	if cols == 0 {
		return 0, 0, nil
	}

	rows := input.shape.Len() / cols
	if rows > math.MaxUint32 || cols > math.MaxUint32 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	return rows, cols, nil
}
