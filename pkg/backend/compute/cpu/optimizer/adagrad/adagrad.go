package adagrad

import (
	stdmath "math"

	prim "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

/*
AdaGrad accumulates squared gradients and scales the learning rate per parameter.
G  += g²
p  -= lr * g / (sqrt(G) + ε)
*/
type AdaGrad struct {
	LR      float64
	Eps     float64
	LRDecay float64 // optional learning-rate decay
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

	g := grads
	if ag.WD != 0 {
		g = make([]float64, n)
		copy(g, grads)
		prim.AddScaledVec(g, params, ag.WD)
	}

	g2 := make([]float64, n)
	prim.MulVec(g2, g, g)
	prim.AddVec(ag.G, g2)

	clr := ag.LR
	if ag.LRDecay != 0 {
		clr /= 1 + float64(ag.step-1)*ag.LRDecay
	}

	denom := make([]float64, n)
	for idx, gsq := range ag.G {
		denom[idx] = stdmath.Sqrt(gsq) + ag.Eps
	}

	update := make([]float64, n)
	prim.DivVec(update, g, denom)

	out := make([]float64, n)
	copy(out, params)
	prim.AddScaledVec(out, update, -clr)

	return out
}

/*
AdaDelta eliminates the global learning rate by tracking both squared gradients
and squared parameter updates with exponential moving averages.
E[g²]  = ρ*E[g²] + (1-ρ)*g²
Δp     = -sqrt(E[Δp²]+ε) / sqrt(E[g²]+ε) * g
E[Δp²] = ρ*E[Δp²] + (1-ρ)*Δp²
*/
type AdaDelta struct {
	Rho    float64
	Eps    float64
	WD     float64
	eg2    []float64 // E[g²]
	edp2   []float64 // E[Δp²]
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

	g := grads
	if ad.WD != 0 {
		g = make([]float64, n)
		copy(g, grads)
		prim.AddScaledVec(g, params, ad.WD)
	}

	g2 := make([]float64, n)
	prim.MulVec(g2, g, g)
	prim.ScaleVec(ad.eg2, ad.Rho)
	prim.AddScaledVec(ad.eg2, g2, 1-ad.Rho)

	delta := make([]float64, n)
	for idx := range delta {
		numSqrt := stdmath.Sqrt(ad.edp2[idx] + ad.Eps)
		denSqrt := stdmath.Sqrt(ad.eg2[idx] + ad.Eps)
		delta[idx] = -(numSqrt / denSqrt) * g[idx]
	}

	dp2 := make([]float64, n)
	prim.MulVec(dp2, delta, delta)
	prim.ScaleVec(ad.edp2, ad.Rho)
	prim.AddScaledVec(ad.edp2, dp2, 1-ad.Rho)

	out := make([]float64, n)
	copy(out, params)
	prim.AddVec(out, delta)

	return out
}
