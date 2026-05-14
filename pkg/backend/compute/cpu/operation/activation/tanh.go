package activation

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Tanh applies the rational-approximation tanh elementwise using SIMD on amd64/arm64.
*/
type Tanh struct{}

func NewTanh() *Tanh {
	return &Tanh{}
}

func (tanh *Tanh) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("activation.tanh"); err != nil {
		return nil, err
	}

	tanhKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
