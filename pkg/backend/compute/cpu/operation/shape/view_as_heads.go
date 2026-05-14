package shape

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
ViewAsHeads reshapes [B, T, D] -> [B, H, T, D/H] in flat row-major layout.
*/
type ViewAsHeads struct{}

func NewViewAsHeads(numHeads ...int) *ViewAsHeads {
	return &ViewAsHeads{}
}

func (viewAsHeads *ViewAsHeads) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("shape.view_as_heads"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) != 3 {
		return nil, fmt.Errorf("shape.view_as_heads: expected rank 3, got %d", len(shape))
	}

	if stateDict.NumHeads <= 0 {
		return nil, fmt.Errorf("shape.view_as_heads: num_heads must be positive, got %d", stateDict.NumHeads)
	}

	batch, tokens, dimension := shape[0], shape[1], shape[2]

	if dimension%stateDict.NumHeads != 0 {
		return nil, fmt.Errorf(
			"shape.view_as_heads: dimension %d not divisible by num_heads %d",
			dimension, stateDict.NumHeads,
		)
	}

	headDim := dimension / stateDict.NumHeads
	stateDict.EnsureOperationOutLen(len(stateDict.Inputs[0]))
	transposeKernel(
		stateDict.Out, stateDict.Inputs[0],
		[]int{batch, tokens, stateDict.NumHeads, headDim}, 1, 2,
	)

	return stateDict, nil
}
