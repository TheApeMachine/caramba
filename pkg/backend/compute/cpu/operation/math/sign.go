package math

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Sign applies the elementwise sign function: out[i] = +1 if src[i] > 0, -1 if < 0, 0 if == 0.
Used as the activation in classic discrete Hopfield networks.
shape: [N].
*/
type Sign struct{}

func NewSign() *Sign { return &Sign{} }

func (sign *Sign) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.sign"); err != nil {
		return nil, err
	}

	signKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
