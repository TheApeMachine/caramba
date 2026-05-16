package activation

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Sigmoid applies 1/(1+exp(-x)) elementwise through the active CPU vector path.
*/
type Sigmoid struct{}

func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (sigmoid *Sigmoid) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("activation.sigmoid"); err != nil {
		return nil, err
	}

	sigmoidKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
