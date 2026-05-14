package shape

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Concat concatenates multiple tensors along a single axis.
*/
type Concat struct{}

func NewConcat(dim ...int) *Concat {
	return &Concat{}
}

func (concat *Concat) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("shape.concat"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if stateDict.Dim < 0 || stateDict.Dim >= len(shape) {
		return nil, fmt.Errorf("shape.concat: dim %d out of range rank %d", stateDict.Dim, len(shape))
	}

	rank := len(shape)
	dim := stateDict.Dim

	outer := 1
	for d := 0; d < dim; d++ {
		outer *= shape[d]
	}
	inner := 1
	for d := dim + 1; d < rank; d++ {
		inner *= shape[d]
	}

	dimSize := shape[dim] // dim size of each individual input
	totalDim := dimSize * len(stateDict.Inputs)

	total := outer * totalDim * inner
	stateDict.EnsureOperationOutLen(total)
	concatKernel(stateDict.Out, stateDict.Inputs, outer, dimSize, inner)

	return stateDict, nil
}
