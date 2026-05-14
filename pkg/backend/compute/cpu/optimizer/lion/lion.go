package lion

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
Lion (EvoLved Sign Momentum, Chen et al. 2023). The fused per-element pipeline
runs entirely in AVX2/SSE2/NEON assembly: interpolate, sign, weight-decay,
parameter step, momentum update — all in one kernel.

	update = sign(β1*m + (1-β1)*g)
	p     -= lr * (update + wd*p)
	m      = β2*m + (1-β2)*g
*/
type Lion struct {
}

func NewLion() *Lion {
	return &Lion{}
}

func (lion *Lion) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("lion"); err != nil {
		return nil, err
	}

	lionStep(
		stateDict.Out,
		stateDict.M,
		stateDict.Params,
		stateDict.Grads,
		stateDict.LR,
		stateDict.Beta1,
		stateDict.Beta2,
		stateDict.WD,
	)

	return stateDict, nil
}
