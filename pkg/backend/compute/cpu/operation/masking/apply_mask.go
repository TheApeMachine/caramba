package masking

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
ApplyMask applies an additive mask to attention scores.

Convention: mask[i] = 0.0 to attend, -Inf to block.
output[i] = scores[i] + mask[i]
*/
type ApplyMask struct{}

func NewApplyMask() *ApplyMask { return &ApplyMask{} }

func (applyMask *ApplyMask) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("masking.apply", 2); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs[0]) != len(stateDict.Inputs[1]) {
		return nil, fmt.Errorf(
			"masking.apply: scores and mask length mismatch: scores=%d mask=%d",
			len(stateDict.Inputs[0]), len(stateDict.Inputs[1]),
		)
	}

	applyMaskKernel(stateDict.Out, stateDict.Inputs[0], stateDict.Inputs[1])

	return stateDict, nil
}

// applyMaskScalar is the pure-Go fallback.
func applyMaskScalar(dst, scores, mask []float64) {
	for i := range scores {
		dst[i] = scores[i] + mask[i]
	}
}
