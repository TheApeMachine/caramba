package activation

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
SwiGLU expects input of length 2n (gates|values) and returns length n.
Each output[i] = swish(gates[i]) * values[i] = gates[i] * sigmoid(gates[i]) * values[i].
*/
type SwiGLU struct{}

func NewSwiGLU() *SwiGLU {
	return &SwiGLU{}
}

func (swiglu *SwiGLU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("activation.swiglu"); err != nil {
		return nil, err
	}

	input := stateDict.Inputs[0]

	if len(input)%2 != 0 {
		return nil, fmt.Errorf("activation.swiglu: input length must be even, got %d", len(input))
	}

	stateDict.SetOperationOutput(make([]float64, len(input)/2))
	swigluKernel(stateDict.Out, input)

	return stateDict, nil
}
