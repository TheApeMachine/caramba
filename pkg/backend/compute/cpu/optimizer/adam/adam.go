package adam

/*
Adam implements the Adam optimizer (Kingma & Ba, 2015).
m  = β1*m + (1-β1)*g
v  = β2*v + (1-β2)*g²
m̂  = m / (1-β1^t)
v̂  = v / (1-β2^t)
p -= lr * m̂ / (sqrt(v̂) + ε)

The hot path runs through dedicated AVX2/SSE2/NEON kernels that fuse the
moment updates, sqrt, divide, decoupled weight decay (AdamW), and parameter
step inline — no Go-side scalar arithmetic touches per-element data.
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

// NewAdamW is identical to NewAdam; AdamW behaviour is enabled by passing wd > 0.
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

	lrT := biasCorrectedLR(adam.LR, adam.Beta1, adam.Beta2, adam.step)
	out := make([]float64, n)
	adamStep(out, adam.m, adam.v, params, grads, adam.Beta1, adam.Beta2, lrT, adam.Eps, adam.LR*adam.WD)

	return out
}
