package lars

import (
	stdmath "math"

	prim "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

/*
LARS (Layer-wise Adaptive Rate Scaling, You et al. 2017) scales the effective
learning rate per layer by the ratio of parameter norm to gradient norm.

  local_lr = η * ‖p‖ / (‖g‖ + β*‖p‖)
  v = μ*v + local_lr * (g + β*p)
  p -= v
*/
type LARS struct {
	LR        float64 // base learning rate η
	Momentum  float64 // μ
	WD        float64 // weight decay coefficient β
	Eta       float64 // trust coefficient (default 0.001)
	Eps       float64 // numerical stability
	velocity  []float64
}

func NewLARS(lr, momentum, wd, eta, eps float64) *LARS {
	return &LARS{LR: lr, Momentum: momentum, WD: wd, Eta: eta, Eps: eps}
}

func (lars *LARS) Step(params, grads []float64) []float64 {
	n := len(params)

	if lars.velocity == nil {
		lars.velocity = make([]float64, n)
	}

	pNorm := stdmath.Sqrt(prim.L2NormSq(params))
	gNorm := stdmath.Sqrt(prim.L2NormSq(grads))

	localLR := lars.LR
	if pNorm > 0 && gNorm > 0 {
		localLR = lars.Eta * pNorm / (gNorm + lars.WD*pNorm + lars.Eps)
	}

	// effective gradient: g + β*p
	effGrad := make([]float64, n)
	copy(effGrad, grads)
	if lars.WD != 0 {
		prim.AddScaledVec(effGrad, params, lars.WD)
	}

	// v = μ*v + localLR * effGrad
	prim.ScaleVec(lars.velocity, lars.Momentum)
	prim.AddScaledVec(lars.velocity, effGrad, localLR)

	out := make([]float64, n)
	copy(out, params)
	prim.AddScaledVec(out, lars.velocity, -1)

	return out
}

/*
LAMB (Layer-wise Adaptive Moments optimizer for Batch training, You et al. 2019)
combines Adam moment estimates with LARS-style layer-wise trust ratio.

  m = β1*m + (1-β1)*g
  v = β2*v + (1-β2)*g²
  update = m̂ / (sqrt(v̂) + ε) + β*p
  p -= lr * (‖p‖ / ‖update‖) * update
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

func (lamb *LAMB) Step(params, grads []float64) []float64 {
	n := len(params)
	lamb.step++

	if lamb.m == nil {
		lamb.m = make([]float64, n)
		lamb.v = make([]float64, n)
	}

	prim.ScaleVec(lamb.m, lamb.Beta1)
	prim.AddScaledVec(lamb.m, grads, 1-lamb.Beta1)

	prim.ScaleVec(lamb.v, lamb.Beta2)
	g2 := make([]float64, n)
	prim.MulVec(g2, grads, grads)
	prim.AddScaledVec(lamb.v, g2, 1-lamb.Beta2)

	bc1 := 1 - stdmath.Pow(lamb.Beta1, float64(lamb.step))
	bc2 := 1 - stdmath.Pow(lamb.Beta2, float64(lamb.step))

	mHat := make([]float64, n)
	vHat := make([]float64, n)
	for idx := range mHat {
		mHat[idx] = lamb.m[idx] / bc1
		vHat[idx] = lamb.v[idx] / bc2
	}

	// update = mHat / (sqrt(vHat) + ε) + wd*p
	update := make([]float64, n)
	for idx := range update {
		update[idx] = mHat[idx]/(stdmath.Sqrt(vHat[idx])+lamb.Eps) + lamb.WD*params[idx]
	}

	pNorm := stdmath.Sqrt(prim.L2NormSq(params))
	uNorm := stdmath.Sqrt(prim.L2NormSq(update))

	ratio := lamb.LR
	if pNorm > 0 && uNorm > 0 {
		ratio = lamb.LR * pNorm / uNorm
	}

	out := make([]float64, n)
	copy(out, params)
	prim.AddScaledVec(out, update, -ratio)

	return out
}
