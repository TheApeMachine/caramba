//go:build darwin && cgo

package metal

import (
	"errors"
	"math"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func newMetalCausalBinaryConfig(
	first *metalTensor,
	second *metalTensor,
	out *metalTensor,
) (metalCausalBinaryConfig, error) {
	elementDType, err := metalElementDTypeFor(first.dtype)
	if err != nil {
		return metalCausalBinaryConfig{}, err
	}

	return metalCausalBinaryConfig{first: first, second: second, out: out, elementDType: elementDType}, nil
}

func newMetalCausalTernaryConfig(
	first *metalTensor,
	second *metalTensor,
	third *metalTensor,
	out *metalTensor,
) (metalCausalTernaryConfig, error) {
	elementDType, err := metalElementDTypeFor(first.dtype)
	if err != nil {
		return metalCausalTernaryConfig{}, err
	}

	return metalCausalTernaryConfig{
		first: first, second: second, third: third, out: out, elementDType: elementDType,
	}, nil
}

func newMetalCausalScalarConfig(
	first *metalTensor,
	second *metalTensor,
	third *metalTensor,
	out *metalTensor,
) (metalCausalScalarConfig, error) {
	elementDType, err := metalElementDTypeFor(first.dtype)
	if err != nil {
		return metalCausalScalarConfig{}, err
	}

	return metalCausalScalarConfig{
		first: first, second: second, third: third, out: out, elementDType: elementDType,
	}, nil
}

func (config metalCausalBinaryConfig) withDims(
	rows int,
	inner int,
	cols int,
) (metalCausalBinaryConfig, error) {
	if err := requireMetalCausalUint32(rows, inner, cols, config.out.shape.Len()); err != nil {
		return metalCausalBinaryConfig{}, err
	}

	config.rows, config.inner, config.cols = uint32(rows), uint32(inner), uint32(cols)
	config.count = uint32(config.out.shape.Len())
	return config, nil
}

func (config metalCausalBinaryConfig) withCount(count int) (metalCausalBinaryConfig, error) {
	if err := requireMetalCausalUint32(count); err != nil {
		return metalCausalBinaryConfig{}, err
	}

	config.count = uint32(count)
	return config, nil
}

func (config metalCausalTernaryConfig) withDims(
	rows int,
	inner int,
	cols int,
) (metalCausalTernaryConfig, error) {
	if err := requireMetalCausalUint32(rows, inner, cols, config.out.shape.Len()); err != nil {
		return metalCausalTernaryConfig{}, err
	}

	config.rows, config.inner, config.cols = uint32(rows), uint32(inner), uint32(cols)
	config.count = uint32(config.out.shape.Len())
	return config, nil
}

func (config metalCausalTernaryConfig) withCount(count int) (metalCausalTernaryConfig, error) {
	if err := requireMetalCausalUint32(count); err != nil {
		return metalCausalTernaryConfig{}, err
	}

	config.count = uint32(count)
	return config, nil
}

func (config metalCausalScalarConfig) withCount(count int) (metalCausalScalarConfig, error) {
	if err := requireMetalCausalUint32(count); err != nil {
		return metalCausalScalarConfig{}, err
	}

	config.count = uint32(count)
	config.partialCount = uint32(metalCausalPartialCount(count))
	return config, nil
}

func requireMetalCausalSameDTypeAndBridge(tensors ...*metalTensor) error {
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
			return errors.New("metal causal: tensors belong to different Metal backends")
		}
	}

	return nil
}

func requireMetalCausalSameDTypeInt32Bridge(
	first *metalTensor,
	second *metalTensor,
	out *metalTensor,
) error {
	if second.dtype != dtype.Int32 || first.dtype != out.dtype {
		return tensor.ErrDTypeMismatch
	}

	if first.bridge != second.bridge || first.bridge != out.bridge {
		return errors.New("metal causal: tensors belong to different Metal backends")
	}

	return nil
}

func requireMetalCausalUint32(values ...int) error {
	for _, value := range values {
		if value < 0 || int64(value) > math.MaxUint32 {
			return tensor.ErrShapeMismatch
		}
	}

	return nil
}

func mustMetalShape(dims ...int) tensor.Shape {
	shape, err := tensor.NewShape(dims)
	if err != nil {
		panic(err)
	}

	return shape
}
