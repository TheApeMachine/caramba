package adam

import (
	stdmath "math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
AdamW implements Adam with decoupled weight decay.
*/
type AdamW struct{}

func NewAdamW() *AdamW {
	return &AdamW{}
}

func (adamW *AdamW) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("adamw"); err != nil {
		return nil, err
	}

	stateDict.Step++

	lrT := adamwBiasCorrectedLR(stateDict)
	adamwKernel(
		stateDict.Out,
		stateDict.M,
		stateDict.V,
		stateDict.Params,
		stateDict.Grads,
		stateDict.Beta1,
		stateDict.Beta2,
		lrT,
		stateDict.Eps,
		stateDict.LR*stateDict.WD,
	)

	return stateDict, nil
}

func adamwBiasCorrectedLR(stateDict *state.Dict) float64 {
	bc1 := 1 - stdmath.Pow(stateDict.Beta1, float64(stateDict.Step))
	bc2 := 1 - stdmath.Pow(stateDict.Beta2, float64(stateDict.Step))

	return stateDict.LR * stdmath.Sqrt(bc2) / bc1
}
