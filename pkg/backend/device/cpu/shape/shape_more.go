package shape

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
Additional shape kernels found in the original substrate's shape
package: transpose (general permutation), upsample_nearest2d,
view_as_heads, reshape.

The general transpose is the rank-aware sibling of transpose2d: it
takes a permutation vector specifying the new axis order.
*/

/*
runTranspose applies a permutation vector to a tensor. The
permutation [p_0, p_1, ..., p_{R-1}] specifies that output axis i
maps to input axis p_i.
*/
func runTranspose(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].Float32Native()
	permutation, _ := args[1].Int32Native()
	out, _ := args[2].Float32Native()

	inDims := args[0].Shape().Dims()
	rank := len(inDims)

	if len(permutation) != rank || len(out) != len(input) {
		return tensor.ErrShapeMismatch
	}

	outDims := make([]int, rank)

	for outAxis, inAxis := range permutation {
		if int(inAxis) < 0 || int(inAxis) >= rank {
			return tensor.ErrShapeMismatch
		}

		outDims[outAxis] = inDims[inAxis]
	}

	inStrides := computeRowMajorStrides(inDims)
	outStrides := computeRowMajorStrides(outDims)

	for flatIndex := range input {
		inCoords := flatToCoords(flatIndex, inDims, inStrides)
		outCoords := make([]int, rank)

		for outAxis, inAxis := range permutation {
			outCoords[outAxis] = inCoords[inAxis]
		}

		outFlat := coordsToFlat(outCoords, outStrides)
		out[outFlat] = input[flatIndex]
	}

	return nil
}

func computeRowMajorStrides(dims []int) []int {
	strides := make([]int, len(dims))

	if len(dims) == 0 {
		return strides
	}

	strides[len(dims)-1] = 1

	for index := len(dims) - 2; index >= 0; index-- {
		strides[index] = strides[index+1] * dims[index+1]
	}

	return strides
}

func flatToCoords(flat int, dims, strides []int) []int {
	coords := make([]int, len(dims))

	for axis := 0; axis < len(dims); axis++ {
		coords[axis] = flat / strides[axis]
		flat %= strides[axis]
	}

	return coords
}

func coordsToFlat(coords, strides []int) int {
	flat := 0

	for axis, coord := range coords {
		flat += coord * strides[axis]
	}

	return flat
}

/*
runUpsampleNearest2D — nearest-neighbor upsampling for a NCHW tensor.
Output spatial dims are determined by the output tensor's shape.
*/
func runUpsampleNearest2D(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inView, _ := args[0].Float32Native()
	outView, _ := args[1].Float32Native()

	inDims := args[0].Shape().Dims()
	outDims := args[1].Shape().Dims()

	if len(inDims) != 4 || len(outDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	channels := inDims[1]
	inH := inDims[2]
	inW := inDims[3]
	outH := outDims[2]
	outW := outDims[3]

	if outDims[0] != batch || outDims[1] != channels {
		return tensor.ErrShapeMismatch
	}

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for chIndex := 0; chIndex < channels; chIndex++ {
			for outRow := 0; outRow < outH; outRow++ {
				inRow := outRow * inH / outH

				for outCol := 0; outCol < outW; outCol++ {
					inCol := outCol * inW / outW
					inIdx := ((batchIndex*channels+chIndex)*inH+inRow)*inW + inCol
					outIdx := ((batchIndex*channels+chIndex)*outH+outRow)*outW + outCol
					outView[outIdx] = inView[inIdx]
				}
			}
		}
	}

	return nil
}

/*
runViewAsHeads reshapes [batch, seq, heads × headDim] into
[batch, seq, heads, headDim] in place. The output tensor must have
the rank-4 shape; the input has the rank-3 shape.

Args: (input [batch, seq, total], numHeads_scalar [1], output [batch, seq, heads, headDim]).
*/
func runViewAsHeads(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	inView, _ := args[0].Float32Native()
	heads, _ := args[1].Int32Native()
	outView, _ := args[2].Float32Native()

	inDims := args[0].Shape().Dims()
	outDims := args[2].Shape().Dims()

	if len(inDims) != 3 || len(outDims) != 4 || len(heads) < 1 {
		return tensor.ErrShapeMismatch
	}

	numHeads := int(heads[0])

	if numHeads <= 0 || inDims[2]%numHeads != 0 || outDims[2] != numHeads {
		return tensor.ErrShapeMismatch
	}

	if len(inView) != len(outView) {
		return tensor.ErrShapeMismatch
	}

	copy(outView, inView)
	return nil
}

/*
runReshape is a metadata-only reshape — the data layout is preserved
in row-major order, only the partition into dimensions changes. The
output tensor must already have the target shape; the kernel copies
the underlying bytes.
*/
func runReshape(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inView, _ := args[0].Float32Native()
	outView, _ := args[1].Float32Native()

	if len(inView) != len(outView) {
		return tensor.ErrShapeMismatch
	}

	copy(outView, inView)
	return nil
}
