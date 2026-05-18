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

type metalQuantizationConfig struct {
	input *metalTensor
	out   *metalTensor
	count uint32
}

type metalQuantizationOp int

const (
	metalQuantizationInt8Dequant metalQuantizationOp = iota
	metalQuantizationInt4Dequant
	metalQuantizationInt8Quant
)

func runMetalInt8Dequant(input tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalQuantization(input, out, dtype.Int8, dtype.Float32)
	if err != nil {
		return err
	}

	return dispatchMetalQuantization(metalQuantizationInt8Dequant, "int8_dequant", config)
}

func runMetalInt4Dequant(input tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalQuantization(input, out, dtype.Int4, dtype.Float32)
	if err != nil {
		return err
	}

	return dispatchMetalQuantization(metalQuantizationInt4Dequant, "int4_dequant", config)
}

func runMetalInt8Quant(input tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalQuantization(input, out, dtype.Float32, dtype.Int8)
	if err != nil {
		return err
	}

	return dispatchMetalQuantization(metalQuantizationInt8Quant, "int8_quant", config)
}

func dispatchMetalQuantization(
	operation metalQuantizationOp,
	name string,
	config metalQuantizationConfig,
) error {
	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_quantization(
		config.input.bridge.device,
		C.int(operation),
		config.input.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint64_t(token),
		&status,
	)

	return finishMetalQuantizationDispatch(name, token, rc, status)
}

func requireMetalQuantization(
	input tensor.Tensor,
	out tensor.Tensor,
	inputDType dtype.DType,
	outDType dtype.DType,
) (metalQuantizationConfig, error) {
	tensors, err := requireMetalTensors(input, out)
	if err != nil {
		return metalQuantizationConfig{}, err
	}

	if tensors[0].dtype != inputDType || tensors[1].dtype != outDType {
		return metalQuantizationConfig{}, tensor.ErrDTypeMismatch
	}

	if tensors[0].bridge != tensors[1].bridge {
		return metalQuantizationConfig{}, errors.New("metal quantization: tensors belong to different Metal backends")
	}

	if !tensors[0].shape.Equal(tensors[1].shape) || tensors[0].shape.Len() > math.MaxUint32 {
		return metalQuantizationConfig{}, tensor.ErrShapeMismatch
	}

	return metalQuantizationConfig{
		input: tensors[0],
		out:   tensors[1],
		count: uint32(tensors[0].shape.Len()),
	}, nil
}

func finishMetalQuantizationDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
