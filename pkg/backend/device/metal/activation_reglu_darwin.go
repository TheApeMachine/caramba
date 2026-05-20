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

	"github.com/theapemachine/manifesto/tensor"
)

type metalReGLUConfig struct {
	destination *metalTensor
	gate        *metalTensor
	up          *metalTensor
	count       uint32
}

func runMetalReGLU(gate tensor.Tensor, up tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalReGLU(gate, up, out)
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
	rc := C.metal_dispatch_reglu(
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
		err := fmt.Errorf("metal reglu: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func requireMetalReGLU(
	gate tensor.Tensor,
	up tensor.Tensor,
	out tensor.Tensor,
) (metalReGLUConfig, error) {
	tensors, err := requireMetalTensors(gate, up, out)
	if err != nil {
		return metalReGLUConfig{}, err
	}

	gateTensor := tensors[0]
	upTensor := tensors[1]
	destinationTensor := tensors[2]

	if gateTensor.bridge != upTensor.bridge ||
		gateTensor.bridge != destinationTensor.bridge {
		return metalReGLUConfig{}, errors.New(
			"metal reglu: tensors belong to different Metal backends",
		)
	}

	if gateTensor.dtype != upTensor.dtype ||
		gateTensor.dtype != destinationTensor.dtype {
		return metalReGLUConfig{}, tensor.ErrDTypeMismatch
	}

	if !gateTensor.shape.Equal(upTensor.shape) ||
		!gateTensor.shape.Equal(destinationTensor.shape) {
		return metalReGLUConfig{}, tensor.ErrShapeMismatch
	}

	if gateTensor.shape.Len() > math.MaxUint32 {
		return metalReGLUConfig{}, tensor.ErrShapeMismatch
	}

	return metalReGLUConfig{
		destination: destinationTensor,
		gate:        gateTensor,
		up:          upTensor,
		count:       uint32(gateTensor.shape.Len()),
	}, nil
}
