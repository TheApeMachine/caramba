package shape

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Split divides a tensor into equal-sized chunks of SplitSize along Dim and
returns all chunks concatenated into one flat buffer.
*/
type Split struct{}

func NewSplit(args ...int) *Split {
	return &Split{}
}

func (split *Split) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("shape.split"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()
	rank := len(shape)
	dim := stateDict.Dim

	if dim < 0 || dim >= rank {
		return nil, fmt.Errorf("shape.split: dim %d out of range rank %d", dim, rank)
	}

	if stateDict.SplitSize <= 0 {
		return nil, fmt.Errorf("shape.split: split_size must be positive, got %d", stateDict.SplitSize)
	}

	outer := 1
	for d := 0; d < dim; d++ {
		outer *= shape[d]
	}
	inner := 1
	for d := dim + 1; d < rank; d++ {
		inner *= shape[d]
	}

	dimSize := shape[dim]
	if dimSize%stateDict.SplitSize != 0 {
		return nil, fmt.Errorf(
			"shape.split: dim size %d is not divisible by split size %d",
			dimSize, stateDict.SplitSize,
		)
	}

	stateDict.EnsureOperationOutLen(len(stateDict.Inputs[0]))
	splitKernel(
		stateDict.Out, stateDict.Inputs[0],
		outer, dimSize, stateDict.SplitSize, inner,
	)

	return stateDict, nil
}
