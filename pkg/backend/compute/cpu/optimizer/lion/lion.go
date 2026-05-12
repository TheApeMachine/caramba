package lion

/*
Lion (EvoLved Sign Momentum, Chen et al. 2023). The fused per-element pipeline
runs entirely in AVX2/SSE2/NEON assembly: interpolate, sign, weight-decay,
parameter step, momentum update — all in one kernel.

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

	out := make([]float64, n)
	lionStep(out, lion.m, params, grads, lion.LR, lion.Beta1, lion.Beta2, lion.WD)

	return out
}
