package math

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Sin computes sin(x) elementwise via vectorized SIMD assembly
(AVX2/SSE2 on amd64, NEON on arm64) or a Go scalar fallback on
other targets. Each ISA's path uses its own vector instructions on
its own registers — there is no aliasing across ISAs and no shared
body. The algorithm everywhere is Cody-Waite range reduction to
[-pi/4, pi/4] followed by minimax polynomial approximation, matching
the scalar reference (math.Sin) within tight ULP bounds.
*/
type Sin struct{}

func NewSin() *Sin { return &Sin{} }

func (sin *Sin) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.sin"); err != nil {
		return nil, err
	}

	sinKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
