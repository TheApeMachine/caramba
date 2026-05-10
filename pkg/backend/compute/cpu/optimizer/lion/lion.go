package lion

import prim "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"

/*
Lion (EvoLved Sign Momentum, Chen et al. 2023) uses only the sign of the
update, making it memory-efficient (no second moment) and well-suited for
large-scale training.

  update = sign(β1*m + (1-β1)*g)
  p     -= lr * (update + wd*p)
  m      = β2*m + (1-β2)*g
*/
type Lion struct {
	LR    float64
	Beta1 float64
	Beta2 float64
	WD    float64
	m     []float64
}

func NewLion(lr, beta1, beta2, wd float64) *Lion {
	return &Lion{LR: lr, Beta1: beta1, Beta2: beta2, WD: wd}
}

func (lion *Lion) Step(params, grads []float64) []float64 {
	n := len(params)

	if lion.m == nil {
		lion.m = make([]float64, n)
	}

	// interpolated = β1*m + (1-β1)*g
	interp := make([]float64, n)
	copy(interp, lion.m)
	prim.ScaleVec(interp, lion.Beta1)
	prim.AddScaledVec(interp, grads, 1-lion.Beta1)

	// update = sign(interpolated)
	update := make([]float64, n)
	for idx, val := range interp {
		switch {
		case val > 0:
			update[idx] = 1
		case val < 0:
			update[idx] = -1
		}
	}

	// p -= lr * (update + wd*p)
	out := make([]float64, n)
	copy(out, params)
	prim.AddScaledVec(out, update, -lion.LR)
	if lion.WD != 0 {
		prim.AddScaledVec(out, params, -lion.LR*lion.WD)
	}

	// m = β2*m + (1-β2)*g
	prim.ScaleVec(lion.m, lion.Beta2)
	prim.AddScaledVec(lion.m, grads, 1-lion.Beta2)

	return out
}
