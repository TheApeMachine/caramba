package adam

import (
	stdmath "math"

	prim "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

/*
AdaMax uses the infinity norm for the second moment estimate.
m  = β1*m + (1-β1)*g
u  = max(β2*u, |g|)
p -= (lr / (1-β1^t)) * m / u
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
	ax.step++

	if ax.m == nil {
		ax.m = make([]float64, n)
		ax.u = make([]float64, n)
	}

	prim.ScaleVec(ax.m, ax.Beta1)
	prim.AddScaledVec(ax.m, grads, 1-ax.Beta1)

	// u = max(β2*u, |g|)
	prim.ScaleVec(ax.u, ax.Beta2)
	for idx, g := range grads {
		absG := stdmath.Abs(g)
		if absG > ax.u[idx] {
			ax.u[idx] = absG
		}
	}

	lrT := ax.LR / (1 - stdmath.Pow(ax.Beta1, float64(ax.step)))
	denom := make([]float64, n)
	copy(denom, ax.u)
	prim.AddScalarVec(denom, ax.Eps)

	update := make([]float64, n)
	prim.DivVec(update, ax.m, denom)

	out := make([]float64, n)
	copy(out, params)
	prim.AddScaledVec(out, update, -lrT)

	return out
}
