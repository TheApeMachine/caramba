//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation

#include "bridge_darwin.h"
*/
import "C"

import (
	"math"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type metalWeightGraftConfig struct {
	weights   *metalTensor
	injection *metalTensor
	count     uint32
}

func runMetalWeightGraftAddFloat32(weights tensor.Tensor, injection tensor.Tensor) error {
	config, err := requireMetalWeightGraft(weights, injection)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.weights, config.injection)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_weight_graft_add_float32(
		config.weights.bridge.device,
		config.weights.buffer,
		config.injection.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalUtilityDispatch("weight_graft_add_float32", token, rc, status)
}

func requireMetalWeightGraft(
	weights tensor.Tensor,
	injection tensor.Tensor,
) (metalWeightGraftConfig, error) {
	weightTensor, injectionTensor, err := requireMetalUtilityTensors(weights, injection)
	if err != nil {
		return metalWeightGraftConfig{}, err
	}

	if weightTensor.dtype != dtype.Float32 || injectionTensor.dtype != dtype.Float32 {
		return metalWeightGraftConfig{}, tensor.ErrDTypeMismatch
	}

	if !weightTensor.shape.Equal(injectionTensor.shape) {
		return metalWeightGraftConfig{}, tensor.ErrShapeMismatch
	}

	if weightTensor.shape.Len() > math.MaxUint32 {
		return metalWeightGraftConfig{}, tensor.ErrShapeMismatch
	}

	return metalWeightGraftConfig{
		weights:   weightTensor,
		injection: injectionTensor,
		count:     uint32(weightTensor.shape.Len()),
	}, nil
}
