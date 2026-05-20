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

type metalNormConfig struct {
	input        *metalTensor
	scale        *metalTensor
	bias         *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	rows         uint32
	cols         uint32
}

func runMetalLayerNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalNorm(input, scale, bias, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.scale, config.bias)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_layernorm(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.scale.buffer,
		config.bias.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal layernorm: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalRMSNorm(input tensor.Tensor, scale tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalNorm(input, scale, nil, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.input, config.scale)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_rmsnorm(
		config.input.bridge.device,
		C.int(config.elementDType),
		config.input.buffer,
		config.scale.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal rmsnorm: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func requireMetalNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalNormConfig, error) {
	config, err := metalNormTensors(input, scale, bias, out)
	if err != nil {
		return metalNormConfig{}, err
	}

	rows, cols, err := metalNormDims(config.input, config.scale, config.out)
	if err != nil {
		return metalNormConfig{}, err
	}

	if err := requireMetalNormBias(config.bias, cols); err != nil {
		return metalNormConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.input.dtype)
	if err != nil {
		return metalNormConfig{}, err
	}

	config.rows = uint32(rows)
	config.cols = uint32(cols)
	config.elementDType = elementDType
	return config, nil
}

func metalNormTensors(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalNormConfig, error) {
	inputTensor, scaleTensor, outTensor, err := requireMetalNormCoreTensors(input, scale, out)
	if err != nil {
		return metalNormConfig{}, err
	}

	config := metalNormConfig{input: inputTensor, scale: scaleTensor, out: outTensor}
	if bias == nil {
		return config, requireMetalNormSameDevice(config)
	}

	biasTensor, err := requireMetalTensor(bias)
	if err != nil {
		return metalNormConfig{}, err
	}

	config.bias = biasTensor
	return config, requireMetalNormSameDevice(config)
}

func requireMetalNormCoreTensors(
	input tensor.Tensor,
	scale tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	inputTensor, err := requireMetalTensor(input)
	if err != nil {
		return nil, nil, nil, err
	}

	scaleTensor, err := requireMetalTensor(scale)
	if err != nil {
		return nil, nil, nil, err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return nil, nil, nil, err
	}

	return inputTensor, scaleTensor, outTensor, nil
}

func requireMetalNormSameDevice(config metalNormConfig) error {
	if config.input.dtype != config.scale.dtype || config.input.dtype != config.out.dtype {
		return tensor.ErrDTypeMismatch
	}

	if config.bias != nil && config.bias.dtype != config.input.dtype {
		return tensor.ErrDTypeMismatch
	}

	if config.input.bridge != config.scale.bridge || config.input.bridge != config.out.bridge {
		return errors.New("metal normalization: tensors belong to different Metal backends")
	}

	if config.bias != nil && config.bias.bridge != config.input.bridge {
		return errors.New("metal normalization: tensors belong to different Metal backends")
	}

	return nil
}

func metalNormDims(
	input *metalTensor,
	scale *metalTensor,
	out *metalTensor,
) (int, int, error) {
	if !input.shape.Equal(out.shape) {
		return 0, 0, tensor.ErrShapeMismatch
	}

	dims := input.shape.Dims()
	if len(dims) == 0 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	cols := dims[len(dims)-1]
	if cols == 0 {
		return 0, 0, nil
	}

	scaleDims := scale.shape.Dims()
	if len(scaleDims) != 1 || scaleDims[0] != cols {
		return 0, 0, tensor.ErrShapeMismatch
	}

	rows := input.shape.Len() / cols
	if rows > math.MaxUint32 || cols > math.MaxUint32 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	return rows, cols, nil
}

func requireMetalNormBias(bias *metalTensor, cols int) error {
	if bias == nil {
		return nil
	}

	biasDims := bias.shape.Dims()
	if len(biasDims) != 1 || biasDims[0] != cols {
		return tensor.ErrShapeMismatch
	}

	return nil
}
