package adagrad

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
AdaGrad accumulates squared gradients and scales the learning rate per parameter.
G  += g²
p  -= lr * g / (sqrt(G) + ε)

Full pipeline (incl. optional weight decay) is implemented in dedicated
AVX2/SSE2/NEON kernels — no Go-side primitive composition.
*/
type AdaGrad struct {
}

func NewAdaGrad() *AdaGrad {
	return &AdaGrad{}
}

func (ag *AdaGrad) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("adagrad"); err != nil {
		return nil, err
	}

	stateDict.Step++

	clr := stateDict.LR

	if stateDict.LRDecay != 0 {
		clr /= 1 + float64(stateDict.Step-1)*stateDict.LRDecay
	}

	adagradStep(
		stateDict.Out,
		stateDict.V,
		stateDict.Params,
		stateDict.Grads,
		clr,
		stateDict.Eps,
		stateDict.WD,
	)

	return stateDict, nil
}

/*
AdaDelta tracks both squared gradients and squared parameter updates with
exponential moving averages.
E[g²]  = ρ*E[g²] + (1-ρ)*g²
Δp     = -sqrt(E[Δp²]+ε) / sqrt(E[g²]+ε) * g
E[Δp²] = ρ*E[Δp²] + (1-ρ)*Δp²
*/
type AdaDelta struct {
	Rho  float64
	Eps  float64
	WD   float64
	eg2  []float64
	edp2 []float64
}

func NewAdaDelta(rho, eps, wd float64) *AdaDelta {
	return &AdaDelta{Rho: rho, Eps: eps, WD: wd}
}

func (ad *AdaDelta) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("adadelta"); err != nil {
		return nil, err
	}

	n := len(stateDict.Params)

	if ad.eg2 == nil {
		ad.eg2 = make([]float64, n)
		ad.edp2 = make([]float64, n)
	}

	stateDict.Out = make([]float64, n)
	stateDict.X = stateDict.Out
	adadeltaStep(
		stateDict.Out,
		ad.eg2,
		ad.edp2,
		stateDict.Params,
		stateDict.Grads,
		ad.Rho,
		ad.Eps,
		ad.WD,
	)

	return stateDict, nil
}
