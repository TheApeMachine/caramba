package shape

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Reshape returns a flat copy of data[0] with no structural change.
The caller uses TargetShape to interpret the output.
*/
type Reshape struct{}

func NewReshape(targetShape ...[]int) *Reshape {
	return &Reshape{}
}

func (reshape *Reshape) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("shape.reshape"); err != nil {
		return nil, err
	}

	reshapeKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
