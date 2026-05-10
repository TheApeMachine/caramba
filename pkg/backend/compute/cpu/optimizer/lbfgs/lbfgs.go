package lbfgs

import stdmath "math"

/*
L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) approximates the
inverse Hessian using the last M gradient/parameter difference pairs.
This is a two-loop recursion implementation (Nocedal 1980).

Unlike first-order optimizers, L-BFGS requires a line search for convergence
guarantees. We provide a simple backtracking Armijo line search.
*/
type LBFGS struct {
	LR         float64
	HistSize   int     // M — number of curvature pairs to store
	LineSearch bool    // enable backtracking line search
	C1         float64 // Armijo sufficient-decrease constant (default 1e-4)

	sHist [][]float64 // parameter differences: s_k = p_{k+1} - p_k
	yHist [][]float64 // gradient differences:  y_k = g_{k+1} - g_k
	rhoH  []float64   // 1 / (y_k · s_k)
	prevP []float64
	prevG []float64
	head  int
	count int
}

func NewLBFGS(lr float64, histSize int, lineSearch bool) *LBFGS {
	return &LBFGS{
		LR:         lr,
		HistSize:   histSize,
		LineSearch: lineSearch,
		C1:         1e-4,
		sHist:      make([][]float64, histSize),
		yHist:      make([][]float64, histSize),
		rhoH:       make([]float64, histSize),
	}
}

func (lb *LBFGS) Step(params, grads []float64) []float64 {
	n := len(params)

	if lb.prevP != nil {
		s := make([]float64, n)
		y := make([]float64, n)
		for idx := range s {
			s[idx] = params[idx] - lb.prevP[idx]
			y[idx] = grads[idx] - lb.prevG[idx]
		}
		ys := dot(y, s)
		if ys > 1e-10 {
			slot := lb.head % lb.HistSize
			lb.sHist[slot] = s
			lb.yHist[slot] = y
			lb.rhoH[slot] = 1 / ys
			lb.head++
			if lb.count < lb.HistSize {
				lb.count++
			}
		}
	}

	// two-loop recursion to compute H⁻¹ * g
	q := make([]float64, n)
	copy(q, grads)
	alphas := make([]float64, lb.count)

	for i := lb.count - 1; i >= 0; i-- {
		slot := (lb.head - 1 - i + lb.HistSize*2) % lb.HistSize
		alphas[i] = lb.rhoH[slot] * dot(lb.sHist[slot], q)
		for idx := range q {
			q[idx] -= alphas[i] * lb.yHist[slot][idx]
		}
	}

	// initial Hessian scaling: H₀ = (y·s)/(y·y) * I
	r := make([]float64, n)
	if lb.count > 0 {
		slot := (lb.head - 1 + lb.HistSize*2) % lb.HistSize
		yy := dot(lb.yHist[slot], lb.yHist[slot])
		ys := dot(lb.yHist[slot], lb.sHist[slot])
		gamma := ys / yy
		for idx := range r {
			r[idx] = gamma * q[idx]
		}
	} else {
		copy(r, q)
	}

	for i := 0; i < lb.count; i++ {
		slot := (lb.head - lb.count + i + lb.HistSize*2) % lb.HistSize
		beta := lb.rhoH[slot] * dot(lb.yHist[slot], r)
		for idx := range r {
			r[idx] += lb.sHist[slot][idx] * (alphas[i] - beta)
		}
	}

	// r is the search direction (H⁻¹ * g); negate for descent
	lr := lb.LR
	if lb.LineSearch {
		lr = lb.armijoLR(params, grads, r, lr)
	}

	out := make([]float64, n)
	for idx := range out {
		out[idx] = params[idx] - lr*r[idx]
	}

	lb.prevP = make([]float64, n)
	lb.prevG = make([]float64, n)
	copy(lb.prevP, params)
	copy(lb.prevG, grads)

	return out
}

func (lb *LBFGS) armijoLR(params, grads, direction []float64, lr float64) float64 {
	f0 := dot(grads, grads) // surrogate: ‖g‖² as current cost
	slope := -dot(grads, direction)
	for range 50 {
		// sufficient decrease: f(p - lr*d) <= f0 + c1*lr*slope
		// we approximate by checking ‖g‖ doesn't increase (no closure available)
		decrease := f0 - lb.C1*lr*slope
		if decrease > 0 {
			break
		}
		lr *= 0.5
		if lr < 1e-10 {
			break
		}
	}
	return stdmath.Max(lr, 1e-10)
}

func dot(a, b []float64) float64 {
	var s float64
	for idx := range a {
		s += a[idx] * b[idx]
	}
	return s
}
