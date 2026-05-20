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

type metalReductionConfig struct {
	input        *metalTensor
	out          *metalTensor
	scratchA     *metalTensor
	scratchB     *metalTensor
	elementDType metalElementDType
	count        uint32
	partialCount uint32
}

func runMetalReductionKernel(operation metalReductionOp, args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalReduction(operation, args[0], args[1])
}

func runMetalReduction(
	operation metalReductionOp,
	input tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalReduction(input, out)
	if err != nil {
		return err
	}

	if err := config.allocateScratch(); err != nil {
		return err
	}

	token, err := config.beginCompletion()
	if err != nil {
		config.closeScratch()
		return err
	}

	config.closeScratch()

	status := C.MetalStatus{}
	rc := C.metal_dispatch_reduction(
		config.input.bridge.device,
		C.int(operation),
		C.int(config.elementDType),
		config.input.buffer,
		config.scratchA.buffer,
		config.scratchB.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint32_t(config.partialCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalReductionDispatch(token, rc, status)
}

func requireMetalReduction(input tensor.Tensor, out tensor.Tensor) (metalReductionConfig, error) {
	tensors, err := requireMetalTensors(input, out)
	if err != nil {
		return metalReductionConfig{}, err
	}

	inputTensor := tensors[0]
	outTensor := tensors[1]

	if inputTensor.dtype != outTensor.dtype {
		return metalReductionConfig{}, tensor.ErrDTypeMismatch
	}

	if inputTensor.shape.Len() == 0 || outTensor.shape.Len() < 1 {
		return metalReductionConfig{}, tensor.ErrShapeMismatch
	}

	if inputTensor.bridge != outTensor.bridge {
		return metalReductionConfig{}, errors.New("metal reduction: tensors belong to different Metal backends")
	}

	if inputTensor.shape.Len() > math.MaxUint32 {
		return metalReductionConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalReductionConfig{}, err
	}

	count := inputTensor.shape.Len()

	return metalReductionConfig{
		input:        inputTensor,
		out:          outTensor,
		elementDType: elementDType,
		count:        uint32(count),
		partialCount: uint32(metalReductionPartialCount(count)),
	}, nil
}

func (config *metalReductionConfig) allocateScratch() error {
	var err error
	config.scratchA, err = newMetalReductionScratch(config.out.bridge, int(config.partialCount))
	if err != nil {
		return err
	}

	config.scratchB, err = newMetalReductionScratch(config.out.bridge, int(config.partialCount))
	if err != nil {
		_ = config.scratchA.Close()
		return err
	}

	return nil
}

func (config *metalReductionConfig) beginCompletion() (uint64, error) {
	return metalCompletions.BeginMany(
		[]*metalTensor{config.out, config.scratchA, config.scratchB},
		config.input,
	)
}

func (config *metalReductionConfig) closeScratch() {
	_ = config.scratchA.Close()
	_ = config.scratchB.Close()
}

func newMetalReductionScratch(bridge *metalBridge, partialCount int) (*metalTensor, error) {
	shape, err := tensor.NewShape([]int{partialCount})
	if err != nil {
		return nil, err
	}

	return bridge.empty(shape, dtype.Float32)
}

func finishMetalReductionDispatch(token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal reduction: %s", metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
