package math

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

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

	input := stateDict.Inputs[0]

	if input == nil {
		return nil, fmt.Errorf("math.sin: stateDict.Inputs[0] must be non-nil before sinKernel")
	}

	expectedLength, err := sin.inputLength(stateDict)

	if err != nil {
		return nil, err
	}

	if len(input) != expectedLength {
		return nil, fmt.Errorf(
			"math.sin: stateDict.Inputs[0] length %d does not match shape length %d before sinKernel",
			len(input),
			expectedLength,
		)
	}

	if len(input) > 0 && stateDict.Out == nil {
		return nil, fmt.Errorf("math.sin: stateDict.Out must be non-nil before sinKernel")
	}

	if len(stateDict.Out) < len(input) {
		return nil, fmt.Errorf(
			"math.sin: stateDict.Out length %d is smaller than stateDict.Inputs[0] length %d before sinKernel",
			len(stateDict.Out),
			len(input),
		)
	}

	sinKernel(stateDict.Out, input)

	return stateDict, nil
}

func (sin *Sin) inputLength(stateDict *state.Dict) (int, error) {
	shape := stateDict.OperationShape()
	expectedLength := 1

	for dimensionIndex, dimension := range shape {
		if dimension < 0 {
			return 0, fmt.Errorf(
				"math.sin: shape dimension %d is negative: %d",
				dimensionIndex,
				dimension,
			)
		}

		expectedLength *= dimension
	}

	return expectedLength, nil
}
