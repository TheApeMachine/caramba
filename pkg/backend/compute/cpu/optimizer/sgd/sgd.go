package sgd

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
SGD implements stochastic gradient descent. Momentum and Nesterov correction
are executed by the architecture kernel selected for the current CPU.
*/
type SGD struct {
}

func NewSGD() *SGD {
	return &SGD{}
}

func (sgd *SGD) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("sgd"); err != nil {
		return nil, err
	}

	if stateDict.Momentum == 0 {
		sgdVanilla(
			stateDict.Out,
			stateDict.Params,
			stateDict.Grads,
			stateDict.LR,
			stateDict.WD,
		)

		return stateDict, nil
	}

	sgdMomentum(
		stateDict.Out,
		stateDict.Params,
		stateDict.Grads,
		stateDict.M,
		stateDict.LR,
		stateDict.WD,
		stateDict.Momentum,
		stateDict.Nesterov,
	)

	return stateDict, nil
}
