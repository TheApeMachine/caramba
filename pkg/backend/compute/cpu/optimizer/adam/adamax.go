package adam

import (
	stdmath "math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
AdaMax uses the infinity norm for the second moment estimate.
m  = β1*m + (1-β1)*g
u  = max(β2*u, |g|)
p -= (lr / (1-β1^t)) * m / (u + ε)
*/
type AdaMax struct {
}

func NewAdaMax() *AdaMax {
	return &AdaMax{}
}

func (adaMax *AdaMax) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("adamax"); err != nil {
		return nil, err
	}

	stateDict.Step++

	lrT := stateDict.LR /
		(1 - stdmath.Pow(stateDict.Beta1, float64(stateDict.Step)))
	adamaxKernel(
		stateDict.Out,
		stateDict.M,
		stateDict.V,
		stateDict.Params,
		stateDict.Grads,
		stateDict.Beta1,
		stateDict.Beta2,
		lrT,
		stateDict.Eps,
	)

	return stateDict, nil
}
