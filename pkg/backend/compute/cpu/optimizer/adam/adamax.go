package adam

import stdmath "math"

/*
AdaMax uses the infinity norm for the second moment estimate.
m  = β1*m + (1-β1)*g
u  = max(β2*u, |g|)
p -= (lr / (1-β1^t)) * m / (u + ε)
*/
type AdaMax struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	Eps   float64
	m, u  []float64
	step  int
}

func NewAdaMax(lr, beta1, beta2, eps float64) *AdaMax {
	return &AdaMax{LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps}
}

func (ax *AdaMax) Step(params, grads []float64) []float64 {
	n := len(params)

	if ax.m == nil || len(ax.m) != n {
		ax.m = make([]float64, n)
		ax.u = make([]float64, n)
		ax.step = 0
	}

	ax.step++

	lrT := ax.LR / (1 - stdmath.Pow(ax.Beta1, float64(ax.step)))
	out := make([]float64, n)
	adamaxStep(out, ax.m, ax.u, params, grads, ax.Beta1, ax.Beta2, lrT, ax.Eps)

	return out
}
