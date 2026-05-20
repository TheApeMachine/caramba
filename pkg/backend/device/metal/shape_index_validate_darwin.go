//go:build darwin && cgo

package metal

import (
	"encoding/binary"
	"errors"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func requireMetalGather(
	source tensor.Tensor,
	indices tensor.Tensor,
	out tensor.Tensor,
) (metalShapeIndexedConfig, error) {
	config, err := requireMetalShapeIndexed(source, indices, nil, out)
	if err != nil {
		return metalShapeIndexedConfig{}, err
	}

	sourceDims := config.first.shape.Dims()
	outDims := config.out.shape.Dims()
	if len(sourceDims) != 2 || len(outDims) != 2 ||
		outDims[0] != config.second.shape.Len() || outDims[1] != sourceDims[1] {
		return metalShapeIndexedConfig{}, tensor.ErrShapeMismatch
	}

	return config.withShapeIndexDims(sourceDims[0], sourceDims[1], outDims[0])
}

func requireMetalScatter(
	target tensor.Tensor,
	indices tensor.Tensor,
	updates tensor.Tensor,
	out tensor.Tensor,
) (metalShapeIndexedConfig, error) {
	config, err := requireMetalShapeIndexed(target, indices, updates, out)
	if err != nil {
		return metalShapeIndexedConfig{}, err
	}

	targetDims := config.first.shape.Dims()
	updateDims := config.third.shape.Dims()
	if len(targetDims) != 2 || len(updateDims) != 2 ||
		!config.first.shape.Equal(config.out.shape) ||
		updateDims[0] != config.second.shape.Len() ||
		updateDims[1] != targetDims[1] {
		return metalShapeIndexedConfig{}, tensor.ErrShapeMismatch
	}

	return config.withShapeIndexDims(targetDims[0], targetDims[1], updateDims[0])
}

func requireMetalWhere(
	mask tensor.Tensor,
	positive tensor.Tensor,
	negative tensor.Tensor,
	out tensor.Tensor,
) (metalShapeIndexedConfig, error) {
	config, err := requireMetalBoolSelect(mask, positive, negative, out)
	if err != nil {
		return metalShapeIndexedConfig{}, err
	}

	if !config.second.shape.Equal(config.third.shape) {
		return metalShapeIndexedConfig{}, tensor.ErrShapeMismatch
	}

	return config.withCount(config.out.shape.Len())
}

func requireMetalMaskedFill(
	input tensor.Tensor,
	mask tensor.Tensor,
	scalar tensor.Tensor,
	out tensor.Tensor,
) (metalShapeIndexedConfig, error) {
	config, err := requireMetalBoolSelect(mask, input, scalar, out)
	if err != nil {
		return metalShapeIndexedConfig{}, err
	}

	if !config.first.shape.Equal(config.out.shape) || config.third.shape.Len() < 1 {
		return metalShapeIndexedConfig{}, tensor.ErrShapeMismatch
	}

	return config.withCount(config.out.shape.Len())
}

func requireMetalTranspose(
	input tensor.Tensor,
	permutation tensor.Tensor,
	out tensor.Tensor,
) (metalShapeTransposeConfig, error) {
	inputTensor, permutationTensor, outTensor, err := requireMetalShapePermutation(input, permutation, out)
	if err != nil {
		return metalShapeTransposeConfig{}, err
	}

	permutationValues, err := metalInt32Vector(permutationTensor)
	if err != nil {
		return metalShapeTransposeConfig{}, err
	}

	permutationU32, err := validateTransposePermutation(inputTensor.shape.Dims(), outTensor.shape.Dims(), permutationValues)
	if err != nil {
		return metalShapeTransposeConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(inputTensor.dtype)
	if err != nil {
		return metalShapeTransposeConfig{}, err
	}

	inputStrides, err := uint32Strides(inputTensor.shape.Dims())
	if err != nil {
		return metalShapeTransposeConfig{}, err
	}

	outStrides, err := uint32Strides(outTensor.shape.Dims())
	if err != nil {
		return metalShapeTransposeConfig{}, err
	}

	return metalShapeTransposeConfig{
		input:        inputTensor,
		permutation:  permutationTensor,
		out:          outTensor,
		elementDType: elementDType,
		rank:         uint32(len(permutationValues)),
		count:        uint32(inputTensor.shape.Len()),
		permutationV: permutationU32,
		inputStrides: inputStrides,
		outStrides:   outStrides,
	}, nil
}

func requireMetalShapeIndexed(
	first tensor.Tensor,
	indices tensor.Tensor,
	third tensor.Tensor,
	out tensor.Tensor,
) (metalShapeIndexedConfig, error) {
	firstTensor, secondTensor, outTensor, err := requireMetalShapeIndexBase(first, indices, out)
	if err != nil {
		return metalShapeIndexedConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(firstTensor.dtype)
	if err != nil {
		return metalShapeIndexedConfig{}, err
	}

	config := metalShapeIndexedConfig{
		first:        firstTensor,
		second:       secondTensor,
		out:          outTensor,
		elementDType: elementDType,
	}

	if third == nil {
		return config, nil
	}

	thirdTensor, err := requireMetalTensor(third)
	if err != nil {
		return metalShapeIndexedConfig{}, err
	}

	if thirdTensor.dtype != firstTensor.dtype ||
		thirdTensor.bridge != firstTensor.bridge {
		return metalShapeIndexedConfig{}, tensor.ErrDTypeMismatch
	}

	config.third = thirdTensor
	return config, nil
}

func requireMetalShapeIndexBase(
	first tensor.Tensor,
	indices tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	firstTensor, err := requireMetalTensor(first)
	if err != nil {
		return nil, nil, nil, err
	}

	indexTensor, err := requireMetalTensor(indices)
	if err != nil {
		return nil, nil, nil, err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return nil, nil, nil, err
	}

	if indexTensor.dtype != dtype.Int32 || firstTensor.dtype != outTensor.dtype {
		return nil, nil, nil, tensor.ErrDTypeMismatch
	}

	if firstTensor.bridge != indexTensor.bridge || firstTensor.bridge != outTensor.bridge {
		return nil, nil, nil, errors.New("metal shape: tensors belong to different Metal backends")
	}

	return firstTensor, indexTensor, outTensor, nil
}

func requireMetalBoolSelect(
	mask tensor.Tensor,
	first tensor.Tensor,
	second tensor.Tensor,
	out tensor.Tensor,
) (metalShapeIndexedConfig, error) {
	maskTensor, firstTensor, secondTensor, outTensor, err := requireMetalBoolSelectTensors(mask, first, second, out)
	if err != nil {
		return metalShapeIndexedConfig{}, err
	}

	elementDType, err := metalElementDTypeFor(firstTensor.dtype)
	if err != nil {
		return metalShapeIndexedConfig{}, err
	}

	return metalShapeIndexedConfig{
		first: maskTensor, second: firstTensor, third: secondTensor, out: outTensor,
		elementDType: elementDType,
	}, nil
}

func requireMetalBoolSelectTensors(
	mask tensor.Tensor,
	first tensor.Tensor,
	second tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, *metalTensor, error) {
	tensors, err := requireMetalTensors(mask, first, second, out)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	if tensors[0].dtype != dtype.Bool || tensors[1].dtype != tensors[2].dtype ||
		tensors[1].dtype != tensors[3].dtype {
		return nil, nil, nil, nil, tensor.ErrDTypeMismatch
	}

	if tensors[0].bridge != tensors[1].bridge ||
		tensors[0].bridge != tensors[2].bridge ||
		tensors[0].bridge != tensors[3].bridge {
		return nil, nil, nil, nil, errors.New("metal shape: tensors belong to different Metal backends")
	}

	if !tensors[0].shape.Equal(tensors[1].shape) || !tensors[1].shape.Equal(tensors[3].shape) {
		return nil, nil, nil, nil, tensor.ErrShapeMismatch
	}

	return tensors[0], tensors[1], tensors[2], tensors[3], nil
}

func requireMetalShapePermutation(
	input tensor.Tensor,
	permutation tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	inputTensor, permutationTensor, outTensor, err := requireMetalShapeIndexBase(input, permutation, out)
	if err != nil {
		return nil, nil, nil, err
	}

	if inputTensor.shape.Len() != outTensor.shape.Len() ||
		permutationTensor.shape.Len() != len(inputTensor.shape.Dims()) {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	if err := requireUint32(inputTensor.shape.Len()); err != nil {
		return nil, nil, nil, err
	}

	return inputTensor, permutationTensor, outTensor, nil
}

func (config metalShapeIndexedConfig) withShapeIndexDims(
	rows int,
	inner int,
	cols int,
) (metalShapeIndexedConfig, error) {
	if err := requireUint32(rows); err != nil {
		return metalShapeIndexedConfig{}, err
	}

	if err := requireUint32(inner); err != nil {
		return metalShapeIndexedConfig{}, err
	}

	if err := requireUint32(cols); err != nil {
		return metalShapeIndexedConfig{}, err
	}

	config.rows = uint32(rows)
	config.inner = uint32(inner)
	config.cols = uint32(cols)
	config.count = uint32(config.out.shape.Len())
	return config, nil
}

func (config metalShapeIndexedConfig) withCount(count int) (metalShapeIndexedConfig, error) {
	if err := requireUint32(count); err != nil {
		return metalShapeIndexedConfig{}, err
	}

	config.count = uint32(count)
	return config, nil
}

func metalInt32Vector(input *metalTensor) ([]int32, error) {
	if input.dtype != dtype.Int32 {
		return nil, tensor.ErrDTypeMismatch
	}

	_, bytes, err := input.bridge.download(input)
	if err != nil {
		return nil, err
	}

	if len(bytes)%4 != 0 {
		return nil, tensor.ErrShapeMismatch
	}

	out := make([]int32, len(bytes)/4)
	for index := range out {
		out[index] = int32(binary.LittleEndian.Uint32(bytes[index*4:]))
	}

	return out, nil
}

func validateTransposePermutation(inDims []int, outDims []int, permutation []int32) ([]uint32, error) {
	if len(inDims) != len(outDims) || len(permutation) != len(inDims) {
		return nil, tensor.ErrShapeMismatch
	}

	seen := make([]bool, len(permutation))
	out := make([]uint32, len(permutation))
	for outAxis, inAxis := range permutation {
		inputAxis := int(inAxis)

		if inAxis < 0 || inputAxis >= len(inDims) || seen[inputAxis] {
			return nil, tensor.ErrShapeMismatch
		}

		if outDims[outAxis] != inDims[inputAxis] {
			return nil, tensor.ErrShapeMismatch
		}

		seen[inputAxis] = true
		out[outAxis] = uint32(inputAxis)
	}

	return out, nil
}

func uint32Strides(dims []int) ([]uint32, error) {
	strides := make([]uint32, len(dims))
	if len(dims) == 0 {
		return strides, nil
	}

	stride := 1
	for axis := len(dims) - 1; axis >= 0; axis-- {
		if err := requireUint32(stride); err != nil {
			return nil, err
		}

		strides[axis] = uint32(stride)
		stride *= dims[axis]
	}

	return strides, requireUint32(stride)
}
