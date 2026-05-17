package math

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Cos computes cos(x) elementwise via vectorized SIMD assembly
(AVX2/SSE2 on amd64, NEON on arm64) or a Go scalar fallback on
other targets. Same Cody-Waite range reduction and quadrant
selection as Sin; the only difference is the octant-to-polynomial
table.
*/
type Cos struct{}

func NewCos() *Cos { return &Cos{} }

func (cos *Cos) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.cos"); err != nil {
		return nil, err
	}

	cosKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
