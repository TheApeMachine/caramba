package lbfgs

import stdmath "math"

/*
L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno). Two-loop recursion
operates entirely through AVX2/SSE2/NEON kernels: vector subtract, dot product,
AXPY, scale, and the final parameter step.
*/
type LBFGS struct {
	LR         float64
	HistSize   int
	LineSearch bool
	C1         float64

	sHist [][]float64
	yHist [][]float64
	rhoH  []float64
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
		lbfgsSub(s, params, lb.prevP)
		lbfgsSub(y, grads, lb.prevG)

		ys := lbfgsDot(y, s)

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

	q := make([]float64, n)
	copy(q, grads)
	alphas := make([]float64, lb.count)

	for i := lb.count - 1; i >= 0; i-- {
		slot := (lb.head - 1 - i + lb.HistSize*2) % lb.HistSize
		alphas[i] = lb.rhoH[slot] * lbfgsDot(lb.sHist[slot], q)
		lbfgsAddScaled(q, lb.yHist[slot], -alphas[i])
	}

	r := make([]float64, n)

	if lb.count > 0 {
		slot := (lb.head - 1 + lb.HistSize*2) % lb.HistSize
		yy := lbfgsDot(lb.yHist[slot], lb.yHist[slot])
		ys := lbfgsDot(lb.yHist[slot], lb.sHist[slot])
		gamma := ys / yy
		copy(r, q)
		lbfgsScale(r, gamma)
	} else {
		copy(r, q)
	}

	for i := 0; i < lb.count; i++ {
		slot := (lb.head - lb.count + i + lb.HistSize*2) % lb.HistSize
		beta := lb.rhoH[slot] * lbfgsDot(lb.yHist[slot], r)
		lbfgsAddScaled(r, lb.sHist[slot], alphas[i]-beta)
	}

	lr := lb.LR

	if lb.LineSearch {
		lr = lb.armijoLR(grads, r, lr)
	}

	out := make([]float64, n)
	lbfgsParamStep(out, params, r, lr)

	lb.prevP = make([]float64, n)
	lb.prevG = make([]float64, n)
	copy(lb.prevP, params)
	copy(lb.prevG, grads)

	return out
}

func (lb *LBFGS) armijoLR(grads, direction []float64, lr float64) float64 {
	f0 := lbfgsDot(grads, grads)
	slope := -lbfgsDot(grads, direction)

	for range 50 {
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
