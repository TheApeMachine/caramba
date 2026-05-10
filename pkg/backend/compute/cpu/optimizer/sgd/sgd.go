package sgd

import prim "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"

/*
SGD implements stochastic gradient descent with optional momentum and Nesterov correction.

Update rules:
  vanilla:   p -= lr * (g + wd*p)
  momentum:  v = μ*v - lr*g;  p += v
  nesterov:  v = μ*v - lr*g;  p += μ*v - lr*g
*/
type SGD struct {
	LR       float64
	Momentum float64
	WD       float64
	Nesterov bool
	velocity []float64
}

func NewSGD(lr, momentum, wd float64, nesterov bool) *SGD {
	return &SGD{LR: lr, Momentum: momentum, WD: wd, Nesterov: nesterov}
}

func (sgd *SGD) Step(params, grads []float64) []float64 {
	out := make([]float64, len(params))
	copy(out, params)

	if sgd.WD != 0 {
		prim.AddScaledVec(out, params, -sgd.LR*sgd.WD)
	}

	if sgd.Momentum == 0 {
		prim.AddScaledVec(out, grads, -sgd.LR)
		return out
	}

	if sgd.velocity == nil {
		sgd.velocity = make([]float64, len(params))
	}

	prim.ScaleVec(sgd.velocity, sgd.Momentum)
	prim.AddScaledVec(sgd.velocity, grads, -sgd.LR)

	if sgd.Nesterov {
		prim.AddScaledVec(out, grads, -sgd.LR)
		prim.AddScaledVec(out, sgd.velocity, sgd.Momentum)
	} else {
		prim.AddVec(out, sgd.velocity)
	}

	return out
}
