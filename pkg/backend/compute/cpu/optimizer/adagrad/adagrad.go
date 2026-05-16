package adagrad

import "github.com/theapemachine/caramba/pkg/backend/compute/state"

/*
AdaGrad accumulates squared gradients and scales the learning rate per parameter.
G  += g²
p  -= lr * g / (sqrt(G) + ε)

Weight decay and accumulator mutation are executed inside the architecture
kernel for the selected CPU target.
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
}

func NewAdaDelta() *AdaDelta {
	return &AdaDelta{}
}

func (ad *AdaDelta) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("adadelta"); err != nil {
		return nil, err
	}

	stateDict.EnsureOut()
	stateDict.EnsureBuf()
	adadeltaStep(
		stateDict.Out,
		stateDict.V,
		stateDict.Buf,
		stateDict.Params,
		stateDict.Grads,
		stateDict.Rho,
		stateDict.Eps,
		stateDict.WD,
	)

	return stateDict, nil
}
