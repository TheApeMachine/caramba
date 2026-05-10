package rmsprop

import (
	stdmath "math"

	prim "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

/*
RMSProp maintains a running average of squared gradients.
v  = α*v + (1-α)*g²
p -= lr * g / (sqrt(v) + ε)

With momentum:
buf = μ*buf + lr * g / (sqrt(v) + ε)
p  -= buf
*/
type RMSProp struct {
	LR       float64
	Alpha    float64 // smoothing factor (default 0.99)
	Eps      float64
	Momentum float64
	WD       float64
	Centered bool // subtract mean of gradient (centered RMSProp)
	v, buf   []float64
	grad_avg []float64 // for centered variant
}

func NewRMSProp(lr, alpha, eps, momentum, wd float64, centered bool) *RMSProp {
	return &RMSProp{LR: lr, Alpha: alpha, Eps: eps, Momentum: momentum, WD: wd, Centered: centered}
}

func (rms *RMSProp) Step(params, grads []float64) []float64 {
	n := len(params)

	if rms.v == nil {
		rms.v = make([]float64, n)
		rms.buf = make([]float64, n)
		if rms.Centered {
			rms.grad_avg = make([]float64, n)
		}
	}

	g := grads
	if rms.WD != 0 {
		g = make([]float64, n)
		copy(g, grads)
		prim.AddScaledVec(g, params, rms.WD)
	}

	// v = α*v + (1-α)*g²
	prim.ScaleVec(rms.v, rms.Alpha)
	g2 := make([]float64, n)
	prim.MulVec(g2, g, g)
	prim.AddScaledVec(rms.v, g2, 1-rms.Alpha)

	denom := make([]float64, n)
	copy(denom, rms.v)

	if rms.Centered {
		prim.ScaleVec(rms.grad_avg, rms.Alpha)
		prim.AddScaledVec(rms.grad_avg, g, 1-rms.Alpha)
		gavg2 := make([]float64, n)
		prim.MulVec(gavg2, rms.grad_avg, rms.grad_avg)
		for idx := range denom {
			denom[idx] -= gavg2[idx]
		}
	}

	for idx := range denom {
		denom[idx] = stdmath.Sqrt(denom[idx]) + rms.Eps
	}

	update := make([]float64, n)
	prim.DivVec(update, g, denom)

	out := make([]float64, n)
	copy(out, params)

	if rms.Momentum != 0 {
		prim.ScaleVec(rms.buf, rms.Momentum)
		prim.AddScaledVec(rms.buf, update, rms.LR)
		prim.AddScaledVec(out, rms.buf, -1)
	} else {
		prim.AddScaledVec(out, update, -rms.LR)
	}

	return out
}
