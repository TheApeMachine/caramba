//go:build darwin && cgo

package metal

import (
	"errors"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func requireMetalOptimizer4(
	params tensor.Tensor,
	gradients tensor.Tensor,
	firstState tensor.Tensor,
	secondState tensor.Tensor,
	out tensor.Tensor,
) (metalOptimizer4Config, error) {
	tensors, err := requireMetalTensors(params, gradients, firstState, secondState, out)
	if err != nil {
		return metalOptimizer4Config{}, err
	}

	if err := requireMetalOptimizerStorage(tensors[0], tensors[1], tensors[4]); err != nil {
		return metalOptimizer4Config{}, err
	}

	if err := requireMetalOptimizerState(tensors[0], tensors[2], tensors[3]); err != nil {
		return metalOptimizer4Config{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalOptimizer4Config{}, err
	}

	return metalOptimizer4Config{
		params:       tensors[0],
		gradients:    tensors[1],
		firstState:   tensors[2],
		secondState:  tensors[3],
		out:          tensors[4],
		elementDType: elementDType,
		count:        uint32(tensors[0].shape.Len()),
	}, nil
}

func requireMetalOptimizer3(
	params tensor.Tensor,
	gradients tensor.Tensor,
	state tensor.Tensor,
	out tensor.Tensor,
) (metalOptimizer3Config, error) {
	tensors, err := requireMetalTensors(params, gradients, state, out)
	if err != nil {
		return metalOptimizer3Config{}, err
	}

	if err := requireMetalOptimizerStorage(tensors[0], tensors[1], tensors[3]); err != nil {
		return metalOptimizer3Config{}, err
	}

	if err := requireMetalOptimizerState(tensors[0], tensors[2]); err != nil {
		return metalOptimizer3Config{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalOptimizer3Config{}, err
	}

	return metalOptimizer3Config{
		params:       tensors[0],
		gradients:    tensors[1],
		state:        tensors[2],
		out:          tensors[3],
		elementDType: elementDType,
		count:        uint32(tensors[0].shape.Len()),
	}, nil
}

func requireMetalOptimizer2(
	params tensor.Tensor,
	gradients tensor.Tensor,
	out tensor.Tensor,
) (metalOptimizer2Config, error) {
	tensors, err := requireMetalTensors(params, gradients, out)
	if err != nil {
		return metalOptimizer2Config{}, err
	}

	if err := requireMetalOptimizerStorage(tensors[0], tensors[1], tensors[2]); err != nil {
		return metalOptimizer2Config{}, err
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalOptimizer2Config{}, err
	}

	return metalOptimizer2Config{
		params:       tensors[0],
		gradients:    tensors[1],
		out:          tensors[2],
		elementDType: elementDType,
		count:        uint32(tensors[0].shape.Len()),
	}, nil
}

func requireMetalOptimizerStorage(tensors ...*metalTensor) error {
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
			return errors.New("metal optimizer: tensors belong to different Metal backends")
		}

		if !target.shape.Equal(tensors[0].shape) {
			return tensor.ErrShapeMismatch
		}
	}

	if tensors[0].shape.Len() > math.MaxUint32 {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func requireMetalOptimizerState(params *metalTensor, states ...*metalTensor) error {
	for _, state := range states {
		if state.dtype != dtype.Float32 {
			return tensor.ErrDTypeMismatch
		}

		if state.bridge != params.bridge {
			return errors.New("metal optimizer: tensors belong to different Metal backends")
		}

		if !state.shape.Equal(params.shape) {
			return tensor.ErrShapeMismatch
		}
	}

	return nil
}
