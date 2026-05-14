package math

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
Mul performs elementwise multiply: out[i] = data[0][i] * data[1][i].
*/
type Mul struct{}

func NewMul() *Mul { return &Mul{} }

func (mul *Mul) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperationInputs("math.mul", 2); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs[0]) != len(stateDict.Inputs[1]) {
		return nil, fmt.Errorf(
			"math.mul: input length mismatch: left=%d right=%d",
			len(stateDict.Inputs[0]), len(stateDict.Inputs[1]),
		)
	}

	mulKernel(stateDict.Out, stateDict.Inputs[0], stateDict.Inputs[1])

	return stateDict, nil
}
