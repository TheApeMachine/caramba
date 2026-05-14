package shape

import (
	"fmt"
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Transpose swaps two dimensions of an N-D tensor stored in row-major flat layout.
*/
type Transpose struct{}

func NewTranspose(dims ...int) *Transpose {
	return &Transpose{}
}

func (transpose *Transpose) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("shape.transpose"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()
	rank := len(shape)

	if rank == 0 {
		return nil, fmt.Errorf("shape.transpose: shape is required")
	}

	if stateDict.Dim0 < 0 || stateDict.Dim0 >= rank || stateDict.Dim1 < 0 || stateDict.Dim1 >= rank {
		return nil, fmt.Errorf(
			"shape.transpose: dims out of range dim0=%d dim1=%d rank=%d",
			stateDict.Dim0, stateDict.Dim1, rank,
		)
	}

	elementCount, err := shapeElementCount(shape)

	if err != nil {
		return nil, err
	}

	if len(stateDict.Inputs[0]) != elementCount {
		return nil, fmt.Errorf(
			"shape.transpose: input length %d does not match shape element count %d",
			len(stateDict.Inputs[0]), elementCount,
		)
	}

	stateDict.EnsureOperationOutLen(elementCount)
	transposeKernel(stateDict.Out, stateDict.Inputs[0], shape, stateDict.Dim0, stateDict.Dim1)

	return stateDict, nil
}

func shapeElementCount(shape []int) (int, error) {
	elementCount := int64(1)

	for _, dimension := range shape {
		if dimension < 0 {
			return 0, fmt.Errorf("shape: negative dimension %d", dimension)
		}

		elementCount *= int64(dimension)

		if elementCount > int64(math.MaxInt) {
			return 0, fmt.Errorf("shape: element count overflows int")
		}
	}

	return int(elementCount), nil
}
