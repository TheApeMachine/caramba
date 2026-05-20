//go:build darwin && cgo

package metal

import (
	"errors"
	"math"

	"github.com/theapemachine/manifesto/tensor"
)

func requireMetalPhysicsBinary(
	operation metalPhysicsBinaryOp,
	input tensor.Tensor,
	spacing tensor.Tensor,
	out tensor.Tensor,
) (metalPhysicsBinaryConfig, error) {
	inputTensor, spacingTensor, outTensor, err := requireMetalPhysicsTensors3(input, spacing, out)
	if err != nil {
		return metalPhysicsBinaryConfig{}, err
	}

	rank, dim0, dim1, dim2, err := metalPhysicsDims(operation, inputTensor, outTensor, spacingTensor)
	if err != nil {
		return metalPhysicsBinaryConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalPhysicsBinaryConfig{}, err
	}

	return metalPhysicsBinaryConfig{
		input: inputTensor, spacing: spacingTensor, out: outTensor, elementDType: elementDType,
		count: uint32(inputTensor.shape.Len()), rank: rank, dim0: dim0, dim1: dim1, dim2: dim2,
	}, nil
}

func requireMetalMadelungContinuity(
	density tensor.Tensor,
	velocity tensor.Tensor,
	spacing tensor.Tensor,
	out tensor.Tensor,
) (metalPhysicsTernaryConfig, error) {
	tensors, err := requireMetalTensors(density, velocity, spacing, out)
	if err != nil {
		return metalPhysicsTernaryConfig{}, err
	}

	if err := requireMetalPhysicsSameDType(tensors...); err != nil {
		return metalPhysicsTernaryConfig{}, err
	}

	count := tensors[0].shape.Len()
	if tensors[1].shape.Len() != count || tensors[3].shape.Len() != count ||
		tensors[2].shape.Len() < 1 || count > math.MaxUint32 {
		return metalPhysicsTernaryConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalPhysicsTernaryConfig{}, err
	}

	return metalPhysicsTernaryConfig{
		first: tensors[0], second: tensors[1], third: tensors[2], out: tensors[3],
		elementDType: elementDType, count: uint32(count),
	}, nil
}

func requireMetalPhysicsFFT(
	realIn tensor.Tensor,
	imagIn tensor.Tensor,
	realOut tensor.Tensor,
	imagOut tensor.Tensor,
) (metalPhysicsFFTConfig, error) {
	tensors, err := requireMetalTensors(realIn, imagIn, realOut, imagOut)
	if err != nil {
		return metalPhysicsFFTConfig{}, err
	}

	if err := requireMetalPhysicsSameDType(tensors...); err != nil {
		return metalPhysicsFFTConfig{}, err
	}

	count := tensors[0].shape.Len()
	if tensors[1].shape.Len() != count || tensors[2].shape.Len() != count ||
		tensors[3].shape.Len() != count || count > math.MaxUint32 {
		return metalPhysicsFFTConfig{}, tensor.ErrShapeMismatch
	}

	elementDType, err := metalElementDTypeFor(tensors[0].dtype)
	if err != nil {
		return metalPhysicsFFTConfig{}, err
	}

	return metalPhysicsFFTConfig{
		realIn: tensors[0], imagIn: tensors[1], realOut: tensors[2], imagOut: tensors[3],
		elementDType: elementDType, count: uint32(count),
	}, nil
}

func metalPhysicsDims(
	operation metalPhysicsBinaryOp,
	input *metalTensor,
	out *metalTensor,
	spacing *metalTensor,
) (uint32, uint32, uint32, uint32, error) {
	if out.shape.Len() != input.shape.Len() || spacing.shape.Len() < 1 ||
		input.shape.Len() > math.MaxUint32 {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	dims := input.shape.Dims()
	if operation != metalPhysicsLaplacian {
		return uint32(len(dims)), uint32(input.shape.Len()), 1, 1, nil
	}

	return metalPhysicsLaplacianDims(dims)
}

func metalPhysicsLaplacianDims(dims []int) (uint32, uint32, uint32, uint32, error) {
	if len(dims) < 1 || len(dims) > 3 {
		return 0, 0, 0, 0, tensor.ErrShapeMismatch
	}

	values := [3]uint32{1, 1, 1}
	for index, dim := range dims {
		if dim < 0 || dim > math.MaxUint32 {
			return 0, 0, 0, 0, tensor.ErrShapeMismatch
		}

		values[index] = uint32(dim)
	}

	return uint32(len(dims)), values[0], values[1], values[2], nil
}

func requireMetalPhysicsTensors3(
	first tensor.Tensor,
	second tensor.Tensor,
	third tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(first, second, third)
	if err != nil {
		return nil, nil, nil, err
	}

	if err := requireMetalPhysicsSameDType(tensors...); err != nil {
		return nil, nil, nil, err
	}

	return tensors[0], tensors[1], tensors[2], nil
}

func requireMetalPhysicsSameDType(tensors ...*metalTensor) error {
	for index := 1; index < len(tensors); index++ {
		if tensors[index].dtype != tensors[0].dtype {
			return tensor.ErrDTypeMismatch
		}

		if tensors[index].bridge != tensors[0].bridge {
			return errors.New("metal physics: tensors belong to different Metal backends")
		}
	}

	return nil
}
