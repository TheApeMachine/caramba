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

	computekernels "github.com/theapemachine/caramba/pkg/backend/compute/kernels"
	"github.com/theapemachine/manifesto/tensor"
)

type metalDropoutConfig struct {
	input        *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	count        uint32
	scale        float32
	threshold    uint32
	seed         [4]uint32
}

func runMetalDropout(input tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalDropout(input, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_dropout(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.float(config.scale),
		C.uint32_t(config.threshold),
		C.uint32_t(config.seed[0]),
		C.uint32_t(config.seed[1]),
		C.uint32_t(config.seed[2]),
		C.uint32_t(config.seed[3]),
		C.uint64_t(token),
		&status,
	)

	return finishMetalDropoutDispatch(token, rc, status)
}

func requireMetalDropout(input tensor.Tensor, out tensor.Tensor) (metalDropoutConfig, error) {
	inputTensor, outTensor, err := requireMetalDropoutTensors(input, out)
	if err != nil {
		return metalDropoutConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalDropoutConfig{}, err
	}

	dropoutConfig := computekernels.DefaultDropoutConfig()
	keepProb := float32(1.0 - dropoutConfig.Rate)

	return metalDropoutConfig{
		input: inputTensor, out: outTensor, elementDType: elementDType,
		count: uint32(inputTensor.shape.Len()), scale: 1.0 / keepProb,
		threshold: uint32(float64(keepProb) * (1 << 32)),
		seed:      metalDropoutSeedState(dropoutConfig.Seed),
	}, nil
}

func requireMetalDropoutTensors(input tensor.Tensor, out tensor.Tensor) (*metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(input, out)
	if err != nil {
		return nil, nil, err
	}

	inputTensor := tensors[0]
	outTensor := tensors[1]
	if inputTensor.dtype != outTensor.dtype {
		return nil, nil, tensor.ErrDTypeMismatch
	}

	if inputTensor.bridge != outTensor.bridge {
		return nil, nil, errors.New("metal dropout: tensors belong to different Metal backends")
	}

	if inputTensor.shape.Len() != outTensor.shape.Len() || inputTensor.shape.Len() > math.MaxUint32 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return inputTensor, outTensor, nil
}

func metalDropoutSeedState(seed uint64) [4]uint32 {
	return [4]uint32{
		uint32(seed),
		uint32(seed >> 32),
		uint32(seed ^ 0x9e3779b9),
		uint32((seed >> 32) ^ 0x6c078965),
	}
}

func finishMetalDropoutDispatch(token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal dropout: %s", metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
