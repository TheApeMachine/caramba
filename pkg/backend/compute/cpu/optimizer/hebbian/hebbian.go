package hebbian

import (
	stdmath "math"

	prim "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

/*
Hebbian implements the classic Hebbian learning rule.
In gradient-free form: ΔW = η * post * pre
Here params encodes the weight matrix (row-major), grads encodes pre * post
activations as the outer product signal passed in as "grads".

  W += lr * grads   (Hebb)
*/
type Hebbian struct {
	LR      float64
	MaxNorm float64 // optional weight clipping (0 = disabled)
}

func NewHebbian(lr, maxNorm float64) *Hebbian {
	return &Hebbian{LR: lr, MaxNorm: maxNorm}
}

func (hebb *Hebbian) Step(params, grads []float64) []float64 {
	out := make([]float64, len(params))
	copy(out, params)
	prim.AddScaledVec(out, grads, hebb.LR)

	if hebb.MaxNorm > 0 {
		norm := stdmath.Sqrt(prim.L2NormSq(out))
		if norm > hebb.MaxNorm {
			prim.ScaleVec(out, hebb.MaxNorm/norm)
		}
	}

	return out
}

/*
OjaRule extends Hebb with a decay term that keeps weights on the unit sphere,
implementing PCA via the leading eigenvector.

  ΔW = η * (post*pre - post²*W)
*/
type OjaRule struct {
	LR float64
}

func NewOjaRule(lr float64) *OjaRule {
	return &OjaRule{LR: lr}
}

func (oja *OjaRule) Step(params, grads []float64) []float64 {
	// grads = post*pre signal; params = current weights
	// post² = L2NormSq(params) approximation — caller provides post*pre as grads
	// decay = post² * W, but we need post scalar; approximate via ‖grads‖/‖params‖
	postSq := prim.L2NormSq(grads)

	out := make([]float64, len(params))
	copy(out, params)
	prim.AddScaledVec(out, grads, oja.LR)
	prim.AddScaledVec(out, params, -oja.LR*postSq)

	return out
}

/*
BCM (Bienenstock-Cooper-Munro) implements a sliding threshold for synaptic
modification. Weights strengthen when post > θ and weaken when post < θ,
where θ slides with the mean squared post-synaptic activity.

  θ  = τ⁻¹ * E[post²]
  ΔW = η * post * (post - θ) * pre
*/
type BCM struct {
	LR      float64
	Tau     float64 // time constant for threshold sliding
	theta   float64 // current modification threshold
}

func NewBCM(lr, tau float64) *BCM {
	return &BCM{LR: lr, Tau: tau}
}

func (bcm *BCM) Step(params, grads []float64) []float64 {
	// grads encodes pre*post as outer product signal
	// theta slides toward E[post²]
	postSq := prim.L2NormSq(grads)
	bcm.theta += (postSq - bcm.theta) / bcm.Tau

	factor := bcm.LR * (postSq - bcm.theta)

	out := make([]float64, len(params))
	copy(out, params)
	prim.AddScaledVec(out, grads, factor)

	return out
}
