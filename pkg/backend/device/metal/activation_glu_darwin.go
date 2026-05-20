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
)

type metalGLUConfig struct {
	destination *metalTensor
	gate        *metalTensor
	up          *metalTensor
	count       uint32
}

func runMetalGLU(gate tensor.Tensor, up tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalGLU(gate, up, out)
	if err != nil {
		return err
	}

	if config.count == 0 {
		return nil
	}

	elementDType, err := metalElementDTypeFor(config.gate.dtype)
	if err != nil {
		return err
	}

	token, err := metalCompletions.Begin(config.destination, config.gate, config.up)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_glu(
		config.gate.bridge.device,
		C.int(elementDType),
		config.destination.buffer,
		config.gate.buffer,
		config.up.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal glu: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func requireMetalGLU(
	gate tensor.Tensor,
	up tensor.Tensor,
	out tensor.Tensor,
) (metalGLUConfig, error) {
	tensors, err := requireMetalTensors(gate, up, out)
	if err != nil {
		return metalGLUConfig{}, err
	}

	gateTensor := tensors[0]
	upTensor := tensors[1]
	destinationTensor := tensors[2]

	if gateTensor.bridge != upTensor.bridge ||
		gateTensor.bridge != destinationTensor.bridge {
		return metalGLUConfig{}, errors.New(
			"metal glu: tensors belong to different Metal backends",
		)
	}

	if gateTensor.dtype != upTensor.dtype ||
		gateTensor.dtype != destinationTensor.dtype {
		return metalGLUConfig{}, tensor.ErrDTypeMismatch
	}

	if !gateTensor.shape.Equal(upTensor.shape) ||
		!gateTensor.shape.Equal(destinationTensor.shape) {
		return metalGLUConfig{}, tensor.ErrShapeMismatch
	}

	if gateTensor.shape.Len() > math.MaxUint32 {
		return metalGLUConfig{}, tensor.ErrShapeMismatch
	}

	return metalGLUConfig{
		destination: destinationTensor,
		gate:        gateTensor,
		up:          upTensor,
		count:       uint32(gateTensor.shape.Len()),
	}, nil
}
