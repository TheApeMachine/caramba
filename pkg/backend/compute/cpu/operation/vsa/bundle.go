package vsa

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Bundle superimposes multiple VSA hypervectors by elementwise addition followed by
L2-normalisation. The result has unit norm and lies close to all input vectors
in proportion to their contribution — the core memory-superposition primitive.
shape=[N], data[0..k-1] are the k vectors to bundle → out[N].
*/
type Bundle struct{}

/*
NewBundle instantiates a new Bundle operation.
*/
func NewBundle() *Bundle { return &Bundle{} }

/*
Forward sums all input vectors then L2-normalises the result.
If len(data)==0, returns a zero vector of length n (no normalisation step).
*/
func (bundle *Bundle) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("vsa.bundle: shape is required")
	}

	n := shape[0]

	if n < 0 {
		return nil, fmt.Errorf("vsa.bundle: n must be non-negative, got %d", n)
	}

	for index, vec := range stateDict.Inputs {
		if len(vec) != n {
			return nil, fmt.Errorf(
				"vsa.bundle: input %d length %d does not match n %d",
				index, len(vec), n,
			)
		}
	}

	stateDict.EnsureOperationOutLen(n)

	if len(stateDict.Inputs) == 0 {
		clear(stateDict.Out)

		return stateDict, nil
	}

	clear(stateDict.Out)
	bundleKernel(stateDict.Out, stateDict.Inputs)

	return stateDict, nil
}
