package activation

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
ReLU applies elementwise max(0, x) using SIMD on amd64/arm64.
*/
type ReLU struct{}

func NewReLU() *ReLU {
	return &ReLU{}
}

func (relu *ReLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("activation.relu"); err != nil {
		return nil, err
	}

	reluKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
