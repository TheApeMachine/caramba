package vsa

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Permute applies a deterministic cyclic shift of k positions to encode VSA roles.
Role-binding via permutation is lossless and invertible, making it ideal for
encoding positional structure in hypervector sequences.
shape=[N], data[0]=vector → out[N] shifted by k positions.
Only data[0] is used; the variadic form matches the shared operation Forward(shape, data ...[]float64) convention.
*/
type Permute struct {
}

/*
NewPermute instantiates a stateless Permute operation.
*/
func NewPermute(k ...int) *Permute { return &Permute{} }

/*
Forward applies a cyclic shift of k positions (wrapping).
*/
func (permute *Permute) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("vsa.permute"); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 1 {
		return nil, fmt.Errorf("vsa.permute: shape is required")
	}

	n := shape[0]

	if n < 0 {
		return nil, fmt.Errorf("vsa.permute: n must be non-negative, got %d", n)
	}

	if len(stateDict.Inputs[0]) != n {
		return nil, fmt.Errorf(
			"vsa.permute: input length %d does not match n %d",
			len(stateDict.Inputs[0]), n,
		)
	}

	stateDict.EnsureOperationOutLen(n)
	permuteKernel(stateDict.Out, stateDict.Inputs[0], stateDict.K)

	return stateDict, nil
}
