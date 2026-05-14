package math

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Dropout randomly zeros elements during training with probability P,
scaling survivors by 1/(1-P). During inference it is an identity.
*/
type Dropout struct{}

func NewDropout() *Dropout {
	return &Dropout{}
}

func (dropout *Dropout) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.dropout"); err != nil {
		return nil, err
	}

	dropoutKernel(stateDict.Out, stateDict.Inputs[0], stateDict.P, stateDict.Training)

	return stateDict, nil
}
