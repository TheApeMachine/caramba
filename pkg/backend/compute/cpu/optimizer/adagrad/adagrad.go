package adagrad

/*
AdaGrad accumulates squared gradients and scales the learning rate per parameter.
G  += g²
p  -= lr * g / (sqrt(G) + ε)

Full pipeline (incl. optional weight decay) is implemented in dedicated
AVX2/SSE2/NEON kernels — no Go-side primitive composition.
*/
type AdaGrad struct {
	LR      float64
	Eps     float64
	LRDecay float64
	WD      float64
	G       []float64
	step    int
}

func NewAdaGrad(lr, eps, lrDecay, wd float64) *AdaGrad {
	return &AdaGrad{LR: lr, Eps: eps, LRDecay: lrDecay, WD: wd}
}

func (ag *AdaGrad) Step(params, grads []float64) []float64 {
	n := len(params)
	ag.step++

	if ag.G == nil {
		ag.G = make([]float64, n)
	}

	clr := ag.LR
	if ag.LRDecay != 0 {
		clr /= 1 + float64(ag.step-1)*ag.LRDecay
	}

	out := make([]float64, n)
	adagradStep(out, ag.G, params, grads, clr, ag.Eps, ag.WD)

	return out
}

/*
AdaDelta tracks both squared gradients and squared parameter updates with
exponential moving averages.
E[g²]  = ρ*E[g²] + (1-ρ)*g²
Δp     = -sqrt(E[Δp²]+ε) / sqrt(E[g²]+ε) * g
E[Δp²] = ρ*E[Δp²] + (1-ρ)*Δp²
*/
type AdaDelta struct {
	Rho  float64
	Eps  float64
	WD   float64
	eg2  []float64
	edp2 []float64
}

func NewAdaDelta(rho, eps, wd float64) *AdaDelta {
	return &AdaDelta{Rho: rho, Eps: eps, WD: wd}
}

func (ad *AdaDelta) Step(params, grads []float64) []float64 {
	n := len(params)

	if ad.eg2 == nil {
		ad.eg2 = make([]float64, n)
		ad.edp2 = make([]float64, n)
	}

	out := make([]float64, n)
	adadeltaStep(out, ad.eg2, ad.edp2, params, grads, ad.Rho, ad.Eps, ad.WD)

	return out
}
