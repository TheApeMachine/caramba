package vsa

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
InversePermute reverses the cyclic shift applied by Permute.
Applying InversePermute(k) after Permute(k) recovers the original vector exactly,
making it the left inverse of the role-encoding operator.
shape=[N], data[0]=vector → out[N] shifted by -k positions.
*/
type InversePermute struct {
}

/*
NewInversePermute instantiates a stateless InversePermute operation.
*/
func NewInversePermute(k ...int) *InversePermute { return &InversePermute{} }

/*
Forward reverses the cyclic shift by delegating to Permute with the complementary offset.
*/
func (inversePermute *InversePermute) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("vsa.inverse_permute"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("vsa.inverse_permute: shape is required")
	}

	n := shape[0]

	if n < 0 {
		return nil, fmt.Errorf("vsa.inverse_permute: n must be non-negative, got %d", n)
	}

	if len(stateDict.Inputs[0]) != n {
		return nil, fmt.Errorf(
			"vsa.inverse_permute: input length %d does not match n %d",
			len(stateDict.Inputs[0]), n,
		)
	}

	stateDict.EnsureOperationOutLen(n)
	inversePermuteKernel(stateDict.Out, stateDict.Inputs[0], stateDict.K)

	return stateDict, nil
}
