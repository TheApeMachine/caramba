package activation

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

const (
	seluAlpha      = 1.6732632423543772
	seluScale      = 1.0507009873554805
	seluScaleAlpha = seluAlpha * seluScale
)

/*
SELU applies the self-normalizing scaled exponential linear unit.
*/
type SELU struct{}

/*
NewSELU instantiates a stateless SELU operation.
*/
func NewSELU() *SELU {
	return &SELU{}
}

func (selu *SELU) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if stateDict == nil {
		return nil, fmt.Errorf("activation.selu: state dict is nil")
	}

	if err := stateDict.RequireOperation("activation.selu"); err != nil {
		return nil, err
	}

	if len(stateDict.Inputs) == 0 || len(stateDict.Inputs[0]) == 0 {
		return nil, fmt.Errorf("activation.selu: input[0] is required")
	}

	stateDict.EnsureOperationOutLen(len(stateDict.Inputs[0]))
	seluKernel(stateDict.Out, stateDict.Inputs[0])

	return stateDict, nil
}
