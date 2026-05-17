package shape

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Slice extracts a contiguous range [start:end) along Dim from the input
tensor and writes it to Out, preserving every other dimension. It is
the inverse of Concat — given a joint sequence built by concatenating
two tensors, Slice retrieves either half (or any contiguous window).

Config map keys read through state.Dict:
  - dim   (int): axis to slice on (0-based).
  - start (int): inclusive start index along dim. Negative values are
    rejected.
  - end   (int): exclusive end index along dim. When zero, defaults to
    the input's full size along dim — i.e. "from start to the end".

The kernel is a plain memory copy because slice performs no
arithmetic; SIMD/assembly variants would offer no measurable
improvement over the runtime's copy() builtin, so this op intentionally
omits per-ISA paths (the same pattern shape.last_token uses).
*/
type Slice struct{}

func NewSlice(_ ...int) *Slice {
	return &Slice{}
}

func (slice *Slice) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("shape.slice", 1); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()
	rank := len(shape)

	if rank == 0 {
		return nil, fmt.Errorf("shape.slice: input shape is empty")
	}

	dim := stateDict.Dim

	if dim < 0 || dim >= rank {
		return nil, fmt.Errorf("shape.slice: dim %d out of range rank %d", dim, rank)
	}

	dimSize := shape[dim]

	start := stateDict.SliceStart
	end := stateDict.SliceEnd

	if end == 0 {
		end = dimSize
	}

	if start < 0 || end > dimSize || start >= end {
		return nil, fmt.Errorf(
			"shape.slice: range [%d:%d) is invalid for dim %d size %d",
			start, end, dim, dimSize,
		)
	}

	outer := 1

	for axis := 0; axis < dim; axis++ {
		outer *= shape[axis]
	}

	inner := 1

	for axis := dim + 1; axis < rank; axis++ {
		inner *= shape[axis]
	}

	keep := end - start
	expectedInput := outer * dimSize * inner

	if len(stateDict.Inputs[0]) < expectedInput {
		return nil, fmt.Errorf(
			"shape.slice: input length %d shorter than shape product %d",
			len(stateDict.Inputs[0]), expectedInput,
		)
	}

	stateDict.EnsureOperationOutLen(outer * keep * inner)
	sliceKernel(stateDict.Out, stateDict.Inputs[0], outer, dimSize, inner, start, keep)

	return stateDict, nil
}

func sliceKernel(
	dst []float64, src []float64,
	outer, dimSize, inner int,
	start, keep int,
) {
	rowSize := inner
	outRowSize := keep * rowSize
	srcBatchStride := dimSize * rowSize
	srcOffset := start * rowSize

	for outerIndex := 0; outerIndex < outer; outerIndex++ {
		srcBase := outerIndex*srcBatchStride + srcOffset
		dstBase := outerIndex * outRowSize

		copy(dst[dstBase:dstBase+outRowSize], src[srcBase:srcBase+outRowSize])
	}
}
