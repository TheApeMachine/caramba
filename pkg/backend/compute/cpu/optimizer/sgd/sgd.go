package sgd

/*
SGD implements stochastic gradient descent with optional momentum and Nesterov
correction. The full update is executed entirely by AVX2/SSE2/NEON kernels.
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
	n := len(params)
	out := make([]float64, n)

	if sgd.Momentum == 0 {
		sgdVanilla(out, params, grads, sgd.LR, sgd.WD)
		return out
	}

	if sgd.velocity == nil {
		sgd.velocity = make([]float64, n)
	}

	sgdMomentum(out, params, grads, sgd.velocity, sgd.LR, sgd.WD, sgd.Momentum, sgd.Nesterov)

	return out
}
