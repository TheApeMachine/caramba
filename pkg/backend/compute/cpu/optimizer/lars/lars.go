package lars

import (
	stdmath "math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
LARS (Layer-wise Adaptive Rate Scaling). Trust ratio and effective gradient
are computed in scalar (single layer-wide scalars), then a fused AVX2/SSE2/NEON
kernel writes the per-element update.
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
	LR    float64
	Beta1 float64
	Beta2 float64
	Eps   float64
	WD    float64
	m, v  []float64
	step  int
}

func NewLAMB(lr, beta1, beta2, eps, wd float64) *LAMB {
	return &LAMB{LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps, WD: wd}
}

func (lamb *LAMB) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("lamb"); err != nil {
		return nil, err
	}

	n := len(stateDict.Params)
	lamb.step++

	if lamb.m == nil {
		lamb.m = make([]float64, n)
		lamb.v = make([]float64, n)
	}

	lambEMA(lamb.m, lamb.v, stateDict.Grads, lamb.Beta1, lamb.Beta2)

	bc1Inv := 1.0 / (1 - stdmath.Pow(lamb.Beta1, float64(lamb.step)))
	bc2Inv := 1.0 / (1 - stdmath.Pow(lamb.Beta2, float64(lamb.step)))

	pNorm := stdmath.Sqrt(lambL2NormSq(stateDict.Params))
	uNormSq := lambUpdateNormSq(
		lamb.m, lamb.v, stateDict.Params, bc1Inv, bc2Inv, lamb.Eps, lamb.WD,
	)
	uNorm := stdmath.Sqrt(uNormSq)

	ratio := lamb.LR

	if pNorm > 0 && uNorm > 0 {
		ratio = lamb.LR * pNorm / uNorm
	}

	stateDict.Out = make([]float64, n)
	stateDict.X = stateDict.Out
	lambStep(
		stateDict.Out,
		lamb.m,
		lamb.v,
		stateDict.Params,
		stateDict.Grads,
		ratio,
		bc1Inv,
		bc2Inv,
		lamb.Eps,
		lamb.WD,
	)

	return stateDict, nil
}
