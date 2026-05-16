package lars

import (
	stdmath "math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
LARS (Layer-wise Adaptive Rate Scaling). Trust ratio and effective gradient
are computed once per layer, then the architecture kernel writes the
per-element update.
*/
type LARS struct {
}

func NewLARS() *LARS {
	return &LARS{}
}

func (lars *LARS) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("lars"); err != nil {
		return nil, err
	}

	pNorm := stdmath.Sqrt(lambL2NormSq(stateDict.Params))
	gNorm := stdmath.Sqrt(lambL2NormSq(stateDict.Grads))

	localLR := stateDict.LR

	if pNorm > 0 && gNorm > 0 {
		localLR = stateDict.Eta * pNorm / (gNorm + stateDict.WD*pNorm + stateDict.Eps)
	}

	larsStep(
		stateDict.Out,
		stateDict.M,
		stateDict.Params,
		stateDict.Grads,
		localLR,
		stateDict.Momentum,
		stateDict.WD,
	)

	return stateDict, nil
}

/*
LAMB combines Adam moment estimates with layer-wise trust ratio. EMA, norm
computations, and the parameter step are all in dedicated assembly kernels.
*/
type LAMB struct {
}

func NewLAMB() *LAMB {
	return &LAMB{}
}

func (lamb *LAMB) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("lamb"); err != nil {
		return nil, err
	}

	stateDict.Step++
	stateDict.EnsureOut()
	lambEMA(stateDict.M, stateDict.V, stateDict.Grads, stateDict.Beta1, stateDict.Beta2)

	bc1Inv := 1.0 / (1 - stdmath.Pow(stateDict.Beta1, float64(stateDict.Step)))
	bc2Inv := 1.0 / (1 - stdmath.Pow(stateDict.Beta2, float64(stateDict.Step)))

	pNorm := stdmath.Sqrt(lambL2NormSq(stateDict.Params))
	uNormSq := lambUpdateNormSq(
		stateDict.M, stateDict.V, stateDict.Params, bc1Inv, bc2Inv, stateDict.Eps, stateDict.WD,
	)
	uNorm := stdmath.Sqrt(uNormSq)

	ratio := stateDict.LR

	if pNorm > 0 && uNorm > 0 {
		ratio = stateDict.LR * pNorm / uNorm
	}

	lambStep(
		stateDict.Out,
		stateDict.M,
		stateDict.V,
		stateDict.Params,
		stateDict.Grads,
		ratio,
		bc1Inv,
		bc2Inv,
		stateDict.Eps,
		stateDict.WD,
	)

	return stateDict, nil
}
