package activation

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Gelu evaluates the standard approximate GELU map using SIMD instructions on
amd64/arm64 and a scalar fallback on other platforms.

The formulation matches the assembly implementations: tanh uses the rational
approximation z·(27+z²)/(27+9z²).
*/
type Gelu struct{}

/*
NewGelu returns an Operation that computes approximate GELU elementwise.
*/
func NewGelu() *Gelu {
	return &Gelu{}
}

func (gelu *Gelu) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("activation.gelu"); err != nil {
		return nil, err
	}

	geluKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
