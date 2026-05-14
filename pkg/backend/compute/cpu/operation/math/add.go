package math

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Add performs elementwise addition: out[i] = data[0][i] + data[1][i].
*/
type Add struct{}

func NewAdd() *Add { return &Add{} }

func (add *Add) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("math.add", 2); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs[0]) != len(stateDict.Inputs[1]) {
		return nil, fmt.Errorf(
			"math.add: input length mismatch: left=%d right=%d",
			len(stateDict.Inputs[0]), len(stateDict.Inputs[1]),
		)
	}

	addKernel(stateDict.Out, stateDict.Inputs[0], stateDict.Inputs[1])

	return stateDict, nil
}
