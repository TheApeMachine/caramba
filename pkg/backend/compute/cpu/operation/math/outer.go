package math

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Outer computes the outer product of two vectors: out[i*N+j] = a[i] * b[j].
shape: [M, N] where M = len(data[0]), N = len(data[1]).
Used to accumulate Hebbian weight matrices: W += outer(p, p) for each stored pattern p.
*/
type Outer struct{}

func NewOuter() *Outer { return &Outer{} }

func (outer *Outer) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("math.outer", 2); err != nil {
		return nil, err
	}

	shape := stateDict.OperationShape()

	if len(shape) < 2 {
		return nil, fmt.Errorf("math.outer: len(shape)=%d, need >= 2", len(shape))
	}

	M, N := shape[0], shape[1]

	if len(stateDict.Inputs[0]) != M || len(stateDict.Inputs[1]) != N {
		return nil, fmt.Errorf(
			"math.outer: input length mismatch: len(a)=%d need %d, len(b)=%d need %d",
			len(stateDict.Inputs[0]), M, len(stateDict.Inputs[1]), N,
		)
	}

	stateDict.EnsureOperationOutLen(M * N)
	outerKernel(stateDict.Out, stateDict.Inputs[0], stateDict.Inputs[1], M, N)

	return stateDict, nil
}
