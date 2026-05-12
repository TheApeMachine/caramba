package lars

import stdmath "math"

/*
LARS (Layer-wise Adaptive Rate Scaling). Trust ratio and effective gradient
are computed in scalar (single layer-wide scalars), then a fused AVX2/SSE2/NEON
kernel writes the per-element update.
*/
type LARS struct {
	LR       float64
	Momentum float64
	WD       float64
	Eta      float64
	Eps      float64
	velocity []float64
}

func NewLARS(lr, momentum, wd, eta, eps float64) *LARS {
	return &LARS{LR: lr, Momentum: momentum, WD: wd, Eta: eta, Eps: eps}
}

func (lars *LARS) Step(params, grads []float64) []float64 {
	n := len(params)

	if lars.velocity == nil {
		lars.velocity = make([]float64, n)
	}

	pNorm := stdmath.Sqrt(lambL2NormSq(params))
	gNorm := stdmath.Sqrt(lambL2NormSq(grads))

	localLR := lars.LR

	if pNorm > 0 && gNorm > 0 {
		localLR = lars.Eta * pNorm / (gNorm + lars.WD*pNorm + lars.Eps)
	}

	out := make([]float64, n)
	larsStep(out, lars.velocity, params, grads, localLR, lars.Momentum, lars.WD)

	return out
}

/*
LAMB combines Adam moment estimates with layer-wise trust ratio. EMA, norm
computations, and the parameter step are all in dedicated assembly kernels.
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

	lambEMA(lamb.m, lamb.v, grads, lamb.Beta1, lamb.Beta2)

	bc1Inv := 1.0 / (1 - stdmath.Pow(lamb.Beta1, float64(lamb.step)))
	bc2Inv := 1.0 / (1 - stdmath.Pow(lamb.Beta2, float64(lamb.step)))

	pNorm := stdmath.Sqrt(lambL2NormSq(params))
	uNormSq := lambUpdateNormSq(lamb.m, lamb.v, params, bc1Inv, bc2Inv, lamb.Eps, lamb.WD)
	uNorm := stdmath.Sqrt(uNormSq)

	ratio := lamb.LR

	if pNorm > 0 && uNorm > 0 {
		ratio = lamb.LR * pNorm / uNorm
	}

	out := make([]float64, n)
	lambStep(out, lamb.m, lamb.v, params, grads, ratio, bc1Inv, bc2Inv, lamb.Eps, lamb.WD)

	return out
}
