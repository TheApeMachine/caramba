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

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

type metalHebbianConfig struct {
	weights      *metalTensor
	post         *metalTensor
	pre          *metalTensor
	out          *metalTensor
	elementDType metalElementDType
	postCount    uint32
	preCount     uint32
}

func runMetalHebbianStep(
	weights tensor.Tensor,
	post tensor.Tensor,
	pre tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalHebbian(weights, post, pre, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	token, err := metalCompletions.Begin(config.out, config.weights, config.post, config.pre)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_hebbian_step(
		config.weights.bridge.device,
		C.int(config.elementDType),
		config.weights.buffer,
		config.post.buffer,
		config.pre.buffer,
		config.out.buffer,
		C.uint32_t(config.postCount),
		C.uint32_t(config.preCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalOptimizerDispatch("hebbian_step", token, rc, status)
}

func runMetalLARSStep(
	params tensor.Tensor,
	gradients tensor.Tensor,
	momentum tensor.Tensor,
	out tensor.Tensor,
) error {
	config, err := requireMetalOptimizer3(params, gradients, momentum, out)
	if err != nil {
		return err
	}

	if config.out.shape.Len() == 0 {
		return nil
	}

	scratch, groupCount, err := larsScratch(config)
	if err != nil {
		return err
	}
	defer func() {
		_ = scratch.Close()
	}()

	token, err := metalCompletions.BeginMany(
		[]*metalTensor{config.out, scratch},
		config.params, config.gradients, config.state,
	)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	rc := C.metal_dispatch_lars_step(
		config.params.bridge.device,
		C.int(config.elementDType),
		config.params.buffer,
		config.gradients.buffer,
		config.state.buffer,
		scratch.buffer,
		config.out.buffer,
		C.uint32_t(config.count),
		C.uint32_t(groupCount),
		C.uint64_t(token),
		&status,
	)

	return finishMetalOptimizerDispatch("lars_step", token, rc, status)
}

func requireMetalHebbian(
	weights tensor.Tensor,
	post tensor.Tensor,
	pre tensor.Tensor,
	out tensor.Tensor,
) (metalHebbianConfig, error) {
	tensors, err := requireMetalTensors(weights, post, pre, out)
	if err != nil {
		return metalHebbianConfig{}, err
	}

	if err := requireMetalOptimizerStorage(tensors[0], tensors[3]); err != nil {
		return metalHebbianConfig{}, err
	}

	if tensors[1].dtype != tensors[0].dtype || tensors[2].dtype != tensors[0].dtype {
		return metalHebbianConfig{}, tensor.ErrDTypeMismatch
	}

	if tensors[1].bridge != tensors[0].bridge || tensors[2].bridge != tensors[0].bridge {
		return metalHebbianConfig{}, errors.New("metal optimizer: tensors belong to different Metal backends")
	}

	weightsDims := tensors[0].shape.Dims()
	if len(weightsDims) != 2 || len(tensors[1].shape.Dims()) != 1 || len(tensors[2].shape.Dims()) != 1 {
		return metalHebbianConfig{}, tensor.ErrShapeMismatch
	}

	if tensors[1].shape.Len() != weightsDims[0] || tensors[2].shape.Len() != weightsDims[1] {
		return metalHebbianConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalHebbianConfig{}, err
	}

	return metalHebbianConfig{
		weights:      tensors[0],
		post:         tensors[1],
		pre:          tensors[2],
		out:          tensors[3],
		elementDType: elementDType,
		postCount:    uint32(weightsDims[0]),
		preCount:     uint32(weightsDims[1]),
	}, nil
}

func larsScratch(config metalOptimizer3Config) (*metalTensor, uint32, error) {
	groupCount := uint32((uint64(config.count) + 255) / 256)
	shape, err := tensor.NewShape([]int{int(groupCount) * 2})
	if err != nil {
		return nil, 0, err
	}

	target, err := config.params.bridge.empty(shape, dtype.Float32)
	if err != nil {
		return nil, 0, err
	}

	return target, groupCount, nil
}
