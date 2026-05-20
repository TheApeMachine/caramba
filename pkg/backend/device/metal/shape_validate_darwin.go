//go:build darwin && cgo

package metal

import (
	"encoding/binary"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func requireMetalShapeCopy(
	input tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	inputTensor, outTensor, err := requireMetalShapeSameDType(input, out)
	if err != nil {
		return nil, nil, err
	}

	if inputTensor.bytes != outTensor.bytes {
		return nil, nil, tensor.ErrShapeMismatch
	}

	if err := requireUint32(inputTensor.bytes); err != nil {
		return nil, nil, err
	}

	return inputTensor, outTensor, nil
}

func requireMetalConcat(
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	leftTensor, rightTensor, outTensor, err := requireMetalSameDType3(left, right, out)
	if err != nil {
		return nil, nil, nil, err
	}

	if leftTensor.bytes+rightTensor.bytes != outTensor.bytes {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	if err := requireUint32(outTensor.bytes); err != nil {
		return nil, nil, nil, err
	}

	return leftTensor, rightTensor, outTensor, nil
}

func requireMetalSplit2(
	input tensor.Tensor,
	left tensor.Tensor,
	right tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	inputTensor, leftTensor, rightTensor, err := requireMetalSameDType3(input, left, right)
	if err != nil {
		return nil, nil, nil, err
	}

	if inputTensor.bytes != leftTensor.bytes+rightTensor.bytes {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	if err := requireUint32(inputTensor.bytes); err != nil {
		return nil, nil, nil, err
	}

	return inputTensor, leftTensor, rightTensor, nil
}

func requireMetalSlice(
	input tensor.Tensor,
	dim tensor.Tensor,
	start tensor.Tensor,
	end tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	inputTensor, outTensor, err := requireMetalShapeSameDType(input, out)
	if err != nil {
		return nil, nil, err
	}

	if _, err := requireMetalTensor(dim); err != nil {
		return nil, nil, err
	}

	if _, err := requireMetalTensor(start); err != nil {
		return nil, nil, err
	}

	if _, err := requireMetalTensor(end); err != nil {
		return nil, nil, err
	}

	return inputTensor, outTensor, nil
}

func requireMetalLastToken(
	input tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	inputTensor, outTensor, err := requireMetalShapeSameDType(input, out)
	if err != nil {
		return nil, nil, err
	}

	dims := inputTensor.shape.Dims()

	if len(dims) != 3 || dims[1] <= 0 || outTensor.shape.Len() != dims[0]*dims[2] {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return inputTensor, outTensor, requireUint32(outTensor.bytes)
}

func requireMetalTranspose2D(
	input tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	inputTensor, outTensor, err := requireMetalShapeSameDType(input, out)
	if err != nil {
		return nil, nil, err
	}

	inDims := inputTensor.shape.Dims()
	outDims := outTensor.shape.Dims()

	if len(inDims) != 2 || len(outDims) != 2 ||
		inDims[0] != outDims[1] || inDims[1] != outDims[0] {
		return nil, nil, tensor.ErrShapeMismatch
	}

	if err := requireUint32(inputTensor.shape.Len()); err != nil {
		return nil, nil, err
	}

	return inputTensor, outTensor, requireUint32(outTensor.bytes)
}

func requireMetalUpsampleNearest2D(
	input tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	inputTensor, outTensor, err := requireMetalShapeSameDType(input, out)
	if err != nil {
		return nil, nil, err
	}

	inDims := inputTensor.shape.Dims()
	outDims := outTensor.shape.Dims()

	if len(inDims) != 4 || len(outDims) != 4 ||
		inDims[0] != outDims[0] || inDims[1] != outDims[1] {
		return nil, nil, tensor.ErrShapeMismatch
	}

	if outTensor.shape.Len() > 0 &&
		(inDims[2] <= 0 || inDims[3] <= 0 || outDims[2] <= 0 || outDims[3] <= 0) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	if err := requireUint32(outTensor.shape.Len()); err != nil {
		return nil, nil, err
	}

	return inputTensor, outTensor, requireUint32(outTensor.bytes)
}

func requireMetalShapeSameDType(
	input tensor.Tensor,
	out tensor.Tensor,
) (*metalTensor, *metalTensor, error) {
	inputTensor, err := requireMetalTensor(input)
	if err != nil {
		return nil, nil, err
	}

	outTensor, err := requireMetalTensor(out)
	if err != nil {
		return nil, nil, err
	}

	if inputTensor.dtype != outTensor.dtype {
		return nil, nil, tensor.ErrDTypeMismatch
	}

	if inputTensor.bridge != outTensor.bridge {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return inputTensor, outTensor, nil
}

func requireMetalSameDType3(
	first tensor.Tensor,
	second tensor.Tensor,
	third tensor.Tensor,
) (*metalTensor, *metalTensor, *metalTensor, error) {
	firstTensor, err := requireMetalTensor(first)
	if err != nil {
		return nil, nil, nil, err
	}

	secondTensor, err := requireMetalTensor(second)
	if err != nil {
		return nil, nil, nil, err
	}

	thirdTensor, err := requireMetalTensor(third)
	if err != nil {
		return nil, nil, nil, err
	}

	if firstTensor.dtype != secondTensor.dtype || firstTensor.dtype != thirdTensor.dtype {
		return nil, nil, nil, tensor.ErrDTypeMismatch
	}

	if firstTensor.bridge != secondTensor.bridge || firstTensor.bridge != thirdTensor.bridge {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	return firstTensor, secondTensor, thirdTensor, nil
}

func requireMergeHeadsShape(inputTensor *metalTensor, outTensor *metalTensor) error {
	inDims := inputTensor.shape.Dims()
	outDims := outTensor.shape.Dims()

	if len(inDims) != 4 || len(outDims) != 3 {
		return tensor.ErrShapeMismatch
	}

	if inDims[0] != outDims[0] || inDims[1] != outDims[1] ||
		inDims[2]*inDims[3] != outDims[2] {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func requireSplitHeadsShape(inputTensor *metalTensor, outTensor *metalTensor) error {
	inDims := inputTensor.shape.Dims()
	outDims := outTensor.shape.Dims()

	if len(inDims) != 3 || len(outDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	if inDims[0] != outDims[0] || inDims[1] != outDims[1] ||
		inDims[2] != outDims[2]*outDims[3] {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func requireViewAsHeadsShape(
	inputTensor *metalTensor,
	outTensor *metalTensor,
	headCount int32,
) error {
	inDims := inputTensor.shape.Dims()
	outDims := outTensor.shape.Dims()

	if len(inDims) != 3 || len(outDims) != 4 || headCount <= 0 {
		return tensor.ErrShapeMismatch
	}

	if inDims[0] != outDims[0] || inDims[1] != outDims[1] ||
		int(headCount) != outDims[2] || inDims[2] != outDims[2]*outDims[3] {
		return tensor.ErrShapeMismatch
	}

	return nil
}

func metalInt32Scalar(input tensor.Tensor, bridge *metalBridge) (int32, error) {
	target, err := requireMetalTensor(input)
	if err != nil {
		return 0, err
	}

	if target.bridge != bridge || target.dtype != dtype.Int32 || target.shape.Len() < 1 {
		return 0, tensor.ErrShapeMismatch
	}

	_, bytes, err := bridge.download(target)
	if err != nil {
		return 0, err
	}

	if len(bytes) < 4 {
		return 0, tensor.ErrShapeMismatch
	}

	return int32(binary.LittleEndian.Uint32(bytes)), nil
}

func requireUint32(value int) error {
	if value < 0 || int64(value) > maxMetalUint32 {
		return tensor.ErrShapeMismatch
	}

	return nil
}
