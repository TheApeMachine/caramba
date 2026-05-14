package activation

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
LeakyReLU applies max(alpha*x, x) elementwise using SIMD on amd64/arm64.
*/
type LeakyReLU struct{}

func NewLeakyReLU() *LeakyReLU {
	return &LeakyReLU{}
}

func (leaky *LeakyReLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("activation.leaky_relu"); err != nil {
		return nil, err
	}

	leakyReLUKernel(stateDict.Out, stateDict.Inputs[0], stateDict.Alpha)

	return stateDict, nil
}
