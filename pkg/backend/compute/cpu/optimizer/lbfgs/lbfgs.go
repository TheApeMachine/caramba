package lbfgs

import (
	stdmath "math"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

/*
L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno). Two-loop recursion
operates entirely through AVX2/SSE2/NEON kernels: vector subtract, dot product,
AXPY, scale, and the final parameter step.
*/
type LBFGS struct {
}

func NewLBFGS() *LBFGS {
	return &LBFGS{}
}

func (lb *LBFGS) Step(stateDict *state.Dict) (*state.Dict, error) {
	if err := stateDict.RequireReady("lbfgs"); err != nil {
		return nil, err
	}

	stateDict.EnsureHistory()
	n := len(stateDict.Params)

	if stateDict.PrevParams != nil {
		s := make([]float64, n)
		y := make([]float64, n)
		lbfgsSub(s, stateDict.Params, stateDict.PrevParams)
		lbfgsSub(y, stateDict.Grads, stateDict.PrevGrads)

		ys := lbfgsDot(y, s)

		if ys > 1e-10 {
			slot := stateDict.Head % stateDict.HistSize
			stateDict.SHist[slot] = s
			stateDict.YHist[slot] = y
			stateDict.RhoHist[slot] = 1 / ys
			stateDict.Head++

			if stateDict.Count < stateDict.HistSize {
				stateDict.Count++
			}
		}
	}

	q := make([]float64, n)
	copy(q, stateDict.Grads)
	alphas := make([]float64, stateDict.Count)

	for i := stateDict.Count - 1; i >= 0; i-- {
		slot := (stateDict.Head - 1 - i + stateDict.HistSize*2) % stateDict.HistSize
		alphas[i] = stateDict.RhoHist[slot] * lbfgsDot(stateDict.SHist[slot], q)
		lbfgsAddScaled(q, stateDict.YHist[slot], -alphas[i])
	}

	r := make([]float64, n)

	if stateDict.Count > 0 {
		slot := (stateDict.Head - 1 + stateDict.HistSize*2) % stateDict.HistSize
		yy := lbfgsDot(stateDict.YHist[slot], stateDict.YHist[slot])
		ys := lbfgsDot(stateDict.YHist[slot], stateDict.SHist[slot])
		gamma := ys / yy
		copy(r, q)
		lbfgsScale(r, gamma)
	} else {
		copy(r, q)
	}

	for i := 0; i < stateDict.Count; i++ {
		slot := (stateDict.Head - stateDict.Count + i + stateDict.HistSize*2) % stateDict.HistSize
		beta := stateDict.RhoHist[slot] * lbfgsDot(stateDict.YHist[slot], r)
		lbfgsAddScaled(r, stateDict.SHist[slot], alphas[i]-beta)
	}

	lr := stateDict.LR

	if stateDict.LineSearch {
		lr = lb.armijoLR(stateDict, r, lr)
	}

	lbfgsParamStep(stateDict.Out, stateDict.Params, r, lr)

	stateDict.PrevParams = make([]float64, n)
	stateDict.PrevGrads = make([]float64, n)
	copy(stateDict.PrevParams, stateDict.Params)
	copy(stateDict.PrevGrads, stateDict.Grads)

	return stateDict, nil
}

func (lb *LBFGS) armijoLR(stateDict *state.Dict, direction []float64, lr float64) float64 {
	f0 := lbfgsDot(stateDict.Grads, stateDict.Grads)
	slope := -lbfgsDot(stateDict.Grads, direction)
	c1 := stateDict.C1

	if c1 == 0 {
		c1 = 1e-4
	}

	for range 50 {
		decrease := f0 - c1*lr*slope

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
