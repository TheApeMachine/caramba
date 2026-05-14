package math

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Exp computes exp(x) elementwise via vectorized SIMD assembly (AVX2/SSE2/NEON).
*/
type Exp struct{}

func NewExp() *Exp { return &Exp{} }

func (exp *Exp) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.exp"); err != nil {
		return nil, err
	}

	expKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
