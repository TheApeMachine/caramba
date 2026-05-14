package shape

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
MergeHeads is the inverse of ViewAsHeads.
[B, H, T, head_dim] -> [B, T, H*head_dim]
*/
type MergeHeads struct{}

func NewMergeHeads() *MergeHeads {
	return &MergeHeads{}
}

func (mergeHeads *MergeHeads) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("shape.merge_heads"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) != 4 {
		return nil, fmt.Errorf("shape.merge_heads: expected rank 4, got %d", len(shape))
	}

	stateDict.EnsureOperationOutLen(len(stateDict.Inputs[0]))
	transposeKernel(stateDict.Out, stateDict.Inputs[0], shape, 1, 2)

	return stateDict, nil
}
