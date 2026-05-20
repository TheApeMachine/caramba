package shape

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
runSlice copies a contiguous [start:end) range along one row-major
dimension. Args: (input, dim, start, end, output). When end is zero it
means the full size of the sliced dimension.
*/
func runSlice(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	inView, err := args[0].Float32Native()
	if err != nil {
		return err
	}

	dim, err := int32ScalarTensor(args[1])
	if err != nil {
		return err
	}

	start, err := int32ScalarTensor(args[2])
	if err != nil {
		return err
	}

	end, err := int32ScalarTensor(args[3])
	if err != nil {
		return err
	}

	outView, err := args[4].Float32Native()
	if err != nil {
		return err
	}

	inDims := args[0].Shape().Dims()
	outDims := args[4].Shape().Dims()
	rank := len(inDims)

	if int(dim) < 0 || int(dim) >= rank || len(outDims) != rank {
		return tensor.ErrShapeMismatch
	}

	dimSize := inDims[dim]
	sliceEnd := int(end)

	if sliceEnd == 0 {
		sliceEnd = dimSize
	}

	if int(start) < 0 || sliceEnd < int(start) || sliceEnd > dimSize {
		return tensor.ErrShapeMismatch
	}

	sliceLen := sliceEnd - int(start)

	for axis := 0; axis < rank; axis++ {
		if axis == int(dim) {
			if outDims[axis] != sliceLen {
				return tensor.ErrShapeMismatch
			}

			continue
		}

		if inDims[axis] != outDims[axis] {
			return tensor.ErrShapeMismatch
		}
	}

	expectedOutLen := 1
	for _, extent := range outDims {
		expectedOutLen *= extent
	}

	if len(outView) != expectedOutLen {
		return tensor.ErrShapeMismatch
	}

	outer := 1
	for axis := 0; axis < int(dim); axis++ {
		outer *= inDims[axis]
	}

	inner := 1
	for axis := int(dim) + 1; axis < rank; axis++ {
		inner *= inDims[axis]
	}

	inputStride := dimSize * inner
	blockLen := sliceLen * inner

	for outerIndex := 0; outerIndex < outer; outerIndex++ {
		inOffset := outerIndex*inputStride + int(start)*inner
		outOffset := outerIndex * blockLen

		CopyContiguousFloat32Native(
			outView[outOffset:outOffset+blockLen],
			inView[inOffset:inOffset+blockLen],
		)
	}

	return nil
}

func int32ScalarTensor(value tensor.Tensor) (int32, error) {
	view, err := value.Int32Native()
	if err != nil {
		return 0, err
	}

	if len(view) < 1 {
		return 0, tensor.ErrShapeMismatch
	}

	return view[0], nil
}
