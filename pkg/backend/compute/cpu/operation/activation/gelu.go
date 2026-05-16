package activation

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Gelu evaluates the tanh-form Gaussian Error Linear Unit:

	0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
*/
type Gelu struct{}

/*
NewGelu returns an Operation that computes GELU elementwise.
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
