package activation

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Tanh applies tanh elementwise through the active CPU vector path.
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
