package adam

import (
	stdmath "math"

	prim "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

/*
Adam implements the Adam optimizer (Kingma & Ba, 2015).
m  = β1*m + (1-β1)*g
v  = β2*v + (1-β2)*g²
m̂  = m / (1-β1^t)
v̂  = v / (1-β2^t)
p -= lr * m̂ / (sqrt(v̂) + ε)
*/
type Adam struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	Eps   float64
	WD    float64 // decoupled weight decay (AdamW when non-zero)
	m, v  []float64
	step  int
}

func NewAdam(lr, beta1, beta2, eps, wd float64) *Adam {
	return &Adam{LR: lr, Beta1: beta1, Beta2: beta2, Eps: eps, WD: wd}
}

func NewAdamW(lr, beta1, beta2, eps, wd float64) *Adam {
	return NewAdam(lr, beta1, beta2, eps, wd)
}

func (adam *Adam) Step(params, grads []float64) []float64 {
	n := len(params)
	adam.step++

	if adam.m == nil {
		adam.m = make([]float64, n)
		adam.v = make([]float64, n)
	}

	// m = β1*m + (1-β1)*g
	prim.ScaleVec(adam.m, adam.Beta1)
	prim.AddScaledVec(adam.m, grads, 1-adam.Beta1)

	// v = β2*v + (1-β2)*g²
	prim.ScaleVec(adam.v, adam.Beta2)
	g2 := make([]float64, n)
	prim.MulVec(g2, grads, grads)
	prim.AddScaledVec(adam.v, g2, 1-adam.Beta2)

	// bias correction scalars
	bc1 := 1 - stdmath.Pow(adam.Beta1, float64(adam.step))
	bc2 := 1 - stdmath.Pow(adam.Beta2, float64(adam.step))
	lrT := adam.LR * stdmath.Sqrt(bc2) / bc1

	// denom = sqrt(v) + ε
	denom := make([]float64, n)
	prim.SqrtVec(denom, adam.v)
	prim.AddScalarVec(denom, adam.Eps)

	// update = m / denom
	update := make([]float64, n)
	prim.DivVec(update, adam.m, denom)

	out := make([]float64, n)
	copy(out, params)

	// decoupled weight decay (AdamW)
	if adam.WD != 0 {
		prim.AddScaledVec(out, params, -adam.LR*adam.WD)
	}

	prim.AddScaledVec(out, update, -lrT)

	return out
}
