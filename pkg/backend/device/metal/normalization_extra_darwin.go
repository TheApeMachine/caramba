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

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

const metalDefaultGroupNormGroups = 32

type metalNorm3DConfig struct {
	input        *metalTensor
	scale        *metalTensor
	bias         *metalTensor
	mean         *metalTensor
	variance     *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	batch        uint32
	channels     uint32
	spatial      uint32
	groups       uint32
}

func runMetalGroupNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalGroupNorm(input, scale, bias, out)
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
	rc := C.metal_dispatch_groupnorm(
		config.input.bridge.device, C.int(config.elementDType), config.input.buffer,
		config.scale.buffer, config.bias.buffer, config.out.buffer, C.uint32_t(config.batch),
		C.uint32_t(config.channels), C.uint32_t(config.spatial), C.uint32_t(config.groups),
		C.uint64_t(token), &status,
	)

	return finishMetalNormExtraDispatch("groupnorm", token, rc, status)
}

func runMetalInstanceNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalInstanceNorm(input, scale, bias, out)
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
	rc := C.metal_dispatch_instancenorm(
		config.input.bridge.device, C.int(config.elementDType), config.input.buffer,
		config.scale.buffer, config.bias.buffer, config.out.buffer, C.uint32_t(config.batch),
		C.uint32_t(config.channels), C.uint32_t(config.spatial), C.uint64_t(token), &status,
	)

	return finishMetalNormExtraDispatch("instancenorm", token, rc, status)
}

func runMetalBatchNormEval(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	mean tensor.Tensor,
	variance tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalBatchNormEval(input, scale, bias, mean, variance, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(
		config.out, config.input, config.scale, config.bias, config.mean, config.variance,
	)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_batchnorm_eval(
		config.input.bridge.device, C.int(config.elementDType), config.input.buffer,
		config.scale.buffer, config.bias.buffer, config.mean.buffer, config.variance.buffer,
		config.out.buffer, C.uint32_t(config.batch), C.uint32_t(config.channels),
		C.uint32_t(config.spatial), C.uint64_t(token), &status,
	)

	return finishMetalNormExtraDispatch("batchnorm_eval", token, rc, status)
}

func requireMetalGroupNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalNorm3DConfig, error) {
	config, err := requireMetalAffineNorm3D(input, scale, bias, out)
	if err != nil {
		return metalNorm3DConfig{}, err
	}

	if config.channels%metalDefaultGroupNormGroups != 0 {
		return metalNorm3DConfig{}, tensor.ErrShapeMismatch
	}

	config.groups = metalDefaultGroupNormGroups
	return config, nil
}

func requireMetalInstanceNorm(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalNorm3DConfig, error) {
	return requireMetalAffineNorm3D(input, scale, bias, out)
}

func requireMetalBatchNormEval(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	mean tensor.Tensor,
	variance tensor.Tensor,
	out tensor.Tensor,
) (metalNorm3DConfig, error) {
	config, err := requireMetalAffineNorm3D(input, scale, bias, out)
	if err != nil {
		return metalNorm3DConfig{}, err
	}

	meanTensor, varianceTensor, err := requireMetalNormParamPair(mean, variance, config.input)
	if err != nil {
		return metalNorm3DConfig{}, err
	}

	if err := requireMetalNorm1DParam(meanTensor, int(config.channels)); err != nil {
		return metalNorm3DConfig{}, err
	}

	if err := requireMetalNorm1DParam(varianceTensor, int(config.channels)); err != nil {
		return metalNorm3DConfig{}, err
	}

	config.mean = meanTensor
	config.variance = varianceTensor
	return config, nil
}

func requireMetalAffineNorm3D(
	input tensor.Tensor,
	scale tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalNorm3DConfig, error) {
	tensors, err := requireMetalTensors(input, scale, bias, out)
	if err != nil {
		return metalNorm3DConfig{}, err
	}

	config := metalNorm3DConfig{
		input: tensors[0],
		scale: tensors[1],
		bias:  tensors[2],
		out:   tensors[3],
	}

	if err := requireMetalNormExtraSameDevice(config.input, config.scale, config.bias, config.out); err != nil {
		return metalNorm3DConfig{}, err
	}

	if !config.input.shape.Equal(config.out.shape) {
		return metalNorm3DConfig{}, tensor.ErrShapeMismatch
	}

	if err := config.withDims(); err != nil {
		return metalNorm3DConfig{}, err
	}

	if err := requireMetalNorm1DParam(config.scale, int(config.channels)); err != nil {
		return metalNorm3DConfig{}, err
	}

	if err := requireMetalNorm1DParam(config.bias, int(config.channels)); err != nil {
		return metalNorm3DConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.input.dtype)
	if err != nil {
		return metalNorm3DConfig{}, err
	}

	config.elementDType = elementDType
	return config, nil
}

func (config *metalNorm3DConfig) withDims() error {
	dims := config.input.shape.Dims()
	if len(dims) != 3 {
		return tensor.ErrShapeMismatch
	}

	for _, value := range dims {
		if err := requireUint32(value); err != nil {
			return err
		}
	}

	config.batch = uint32(dims[0])
	config.channels = uint32(dims[1])
	config.spatial = uint32(dims[2])
	return nil
}

func requireMetalNormParamPair(
	mean tensor.Tensor,
	variance tensor.Tensor,
	reference *metalTensor,
) (*metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(mean, variance)
	if err != nil {
		return nil, nil, err
	}

	if err := requireMetalNormExtraSameDevice(reference, tensors[0], tensors[1]); err != nil {
		return nil, nil, err
	}

	return tensors[0], tensors[1], nil
}

func requireMetalNormExtraSameDevice(tensors ...*metalTensor) error {
	if len(tensors) == 0 {
		return tensor.ErrShapeMismatch
	}

	storageDType := tensors[0].dtype
	bridge := tensors[0].bridge
	for _, target := range tensors[1:] {
		if target.dtype != storageDType {
			return tensor.ErrDTypeMismatch
		}

		if target.bridge != bridge {
			return errors.New("metal normalization: tensors belong to different Metal backends")
		}
	}

	return nil
}

func requireMetalNorm1DParam(input *metalTensor, channels int) error {
	dims := input.shape.Dims()
	if len(dims) != 1 || dims[0] != channels {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func finishMetalNormExtraDispatch(name string, token uint64, rc C.int, status C.MetalStatus) error {
	if rc == 0 {
		return nil
	}

	err := fmt.Errorf("metal %s: %s", name, metalStatus("dispatch", status))
	metalCompletions.Fail(token, err)
	return err
}
