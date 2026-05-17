package math

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

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
	if err := stateDict.Err(); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) < 1 {
		return nil, fmt.Errorf("math.cos: at least 1 input(s) required")
	}

	input := stateDict.Inputs[0]

	if stateDict.Out != nil && len(stateDict.Out) < len(input) {
		return nil, fmt.Errorf(
			"math.cos: stateDict.Out length %d is smaller than stateDict.Inputs[0] length %d before cosKernel",
			len(stateDict.Out),
			len(input),
		)
	}

	if err := stateDict.RequireOperation("math.cos"); err != nil {
		return nil, err
	}

	if len(input) > 0 && stateDict.Out == nil {
		return nil, fmt.Errorf("math.cos: stateDict.Out must be non-nil before cosKernel")
	}

	if len(stateDict.Out) < len(input) {
		return nil, fmt.Errorf(
			"math.cos: stateDict.Out length %d is smaller than stateDict.Inputs[0] length %d before cosKernel",
			len(stateDict.Out),
			len(input),
		)
	}

	cosKernel(stateDict.Out, input)

	return stateDict, nil
}
