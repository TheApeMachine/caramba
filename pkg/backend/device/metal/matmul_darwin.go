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

type metalMatMulConfig struct {
	left         *metalTensor
	right        *metalTensor
	bias         *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	rows         uint32
	inner        uint32
	cols         uint32
}

func runMetalMatMul(left tensor.Tensor, right tensor.Tensor, out tensor.Tensor) error {
	config, err := requireMetalMatMul(left, right, nil, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.left, config.right)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_matmul(
		config.left.bridge.device,
		C.int(config.elementDType),
		config.left.buffer,
		config.right.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.inner),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal matmul: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func runMetalMatMulAdd(
	left tensor.Tensor,
	right tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalMatMul(left, right, bias, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.left, config.right, config.bias)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_matmul_add(
		config.left.bridge.device,
		C.int(config.elementDType),
		config.left.buffer,
		config.right.buffer,
		config.bias.buffer,
		config.out.buffer,
		C.uint32_t(config.rows),
		C.uint32_t(config.inner),
		C.uint32_t(config.cols),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal matmul_add: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func requireMetalMatMul(
	left tensor.Tensor,
	right tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalMatMulConfig, error) {
	config, err := metalMatMulTensors(left, right, bias, out)
	if err != nil {
		return metalMatMulConfig{}, err
	}

	rows, inner, cols, err := metalMatMulDims(config.left, config.right, config.out)
	if err != nil {
		return metalMatMulConfig{}, err
	}

	if err := requireMetalMatMulBias(config.bias, cols); err != nil {
		return metalMatMulConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(config.left.dtype)
	if err != nil {
		return metalMatMulConfig{}, err
	}

	config.rows = uint32(rows)
	config.inner = uint32(inner)
	config.cols = uint32(cols)
	config.elementDType = elementDType
	return config, nil
}

func metalMatMulTensors(
	left tensor.Tensor,
	right tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) (metalMatMulConfig, error) {
	leftTensor, rightTensor, outTensor, err := requireMetalMatMulCoreTensors(left, right, out)
	if err != nil {
		return metalMatMulConfig{}, err
	}

	if bias == nil {
		return metalMatMulConfig{left: leftTensor, right: rightTensor, out: outTensor}, nil
	}

	biasTensor, err := requireMetalTensor(bias)
	if err != nil {
		return metalMatMulConfig{}, err
	}

	config := metalMatMulConfig{
		left:  leftTensor,
		right: rightTensor,
		bias:  biasTensor,
		out:   outTensor,
	}
	return config, requireMetalMatMulSameDevice(config)
}

func requireMetalMatMulCoreTensors(
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	leftTensor, err := requireMetalTensor(left)
	if err != nil {
		return nil, nil, nil, err
	}

	rightTensor, err := requireMetalTensor(right)
	if err != nil {
		return nil, nil, nil, err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return nil, nil, nil, err
	}

	config := metalMatMulConfig{left: leftTensor, right: rightTensor, out: outTensor}
	return leftTensor, rightTensor, outTensor, requireMetalMatMulSameDevice(config)
}

func requireMetalMatMulSameDevice(config metalMatMulConfig) error {
	if config.left.dtype != config.right.dtype || config.left.dtype != config.out.dtype {
		return tensor.ErrDTypeMismatch
	}

	if config.bias != nil && config.bias.dtype != config.left.dtype {
		return tensor.ErrDTypeMismatch
	}

	if config.left.bridge != config.right.bridge || config.left.bridge != config.out.bridge {
		return errors.New("metal matmul: tensors belong to different Metal backends")
	}

	if config.bias != nil && config.bias.bridge != config.left.bridge {
		return errors.New("metal matmul: tensors belong to different Metal backends")
	}

	return nil
}

func metalMatMulDims(left *metalTensor, right *metalTensor, out *metalTensor) (int, int, int, error) {
	leftDims := left.shape.Dims()
	rightDims := right.shape.Dims()
	outDims := out.shape.Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 || len(outDims) != 2 {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	if leftDims[1] != rightDims[0] {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	if outDims[0] != leftDims[0] || outDims[1] != rightDims[1] {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	if int64(out.shape.Len()) > math.MaxUint32 {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	return leftDims[0], leftDims[1], rightDims[1], nil
}

func requireMetalMatMulBias(bias *metalTensor, cols int) error {
	if bias == nil {
		return nil
	}

	biasDims := bias.shape.Dims()
	if len(biasDims) != 1 || biasDims[0] != cols {
		return tensor.ErrShapeMismatch
	}

	return nil
}
