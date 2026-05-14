package math

import (
	"fmt"
	gomath "math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
InvSqrtDimScale multiplies each element of data[0] by 1/sqrt(shape[-1]).
This is the standard attention head scale factor.
*/
type InvSqrtDimScale struct{}

func NewInvSqrtDimScale() *InvSqrtDimScale { return &InvSqrtDimScale{} }

func (scale *InvSqrtDimScale) Forward(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireOperation("math.inv_sqrt_dim_scale"); err != nil {
		return nil, err
	}

	dim := stateDict.OperationLastDim()

	if dim <= 0 {
		return nil, fmt.Errorf("math.inv_sqrt_dim_scale: last dimension must be positive, got %d", dim)
	}

	multiplier := 1.0 / gomath.Sqrt(float64(dim))
	invSqrtDimScaleKernel(stateDict.Out, stateDict.Inputs[0], multiplier)

	return stateDict, nil
}
