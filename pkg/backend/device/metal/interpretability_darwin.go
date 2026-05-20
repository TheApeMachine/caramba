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
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type metalActivationSteerConfig struct {
	destination *metalTensor
	base        *metalTensor
	direction   *metalTensor
	coefficient *metalTensor
	count       uint32
}

func runMetalActivationSteerFloat32(
	base tensor.Tensor,
	direction tensor.Tensor,
	coefficient tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalActivationSteer(base, direction, coefficient, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(
		config.destination,
		config.base,
		config.direction,
		config.coefficient,
	)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_activation_steer_float32(
		config.base.bridge.device,
		config.destination.buffer,
		config.base.buffer,
		config.direction.buffer,
		config.coefficient.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalUtilityDispatch("activation_steer_float32", token, rc, status)
}

func requireMetalActivationSteer(
	base tensor.Tensor,
	direction tensor.Tensor,
	coefficient tensor.Tensor,
	out tensor.Tensor,
) (metalActivationSteerConfig, error) {
	tensors, err := requireMetalTensors(base, direction, coefficient, out)
	if err != nil {
		return metalActivationSteerConfig{}, err
	}

	baseTensor := tensors[0]
	directionTensor := tensors[1]
	coefficientTensor := tensors[2]
	destinationTensor := tensors[3]

	if baseTensor.bridge != directionTensor.bridge ||
		baseTensor.bridge != coefficientTensor.bridge ||
		baseTensor.bridge != destinationTensor.bridge {
		return metalActivationSteerConfig{}, errors.New(
			"metal activation steer: tensors belong to different Metal backends",
		)
	}

	if baseTensor.dtype != dtype.Float32 ||
		directionTensor.dtype != dtype.Float32 ||
		coefficientTensor.dtype != dtype.Float32 ||
		destinationTensor.dtype != dtype.Float32 {
		return metalActivationSteerConfig{}, tensor.ErrDTypeMismatch
	}

	if !baseTensor.shape.Equal(directionTensor.shape) ||
		!baseTensor.shape.Equal(destinationTensor.shape) {
		return metalActivationSteerConfig{}, tensor.ErrShapeMismatch
	}

	if coefficientTensor.shape.Len() != 1 {
		return metalActivationSteerConfig{}, tensor.ErrShapeMismatch
	}

	if baseTensor.shape.Len() > math.MaxUint32 {
		return metalActivationSteerConfig{}, tensor.ErrShapeMismatch
	}

	return metalActivationSteerConfig{
		destination: destinationTensor,
		base:        baseTensor,
		direction:   directionTensor,
		coefficient: coefficientTensor,
		count:       uint32(baseTensor.shape.Len()),
	}, nil
}
