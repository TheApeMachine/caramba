package causal

import (
	"fmt"
	"math"
	"sort"
)

/*
FrontdoorAdjustment computes the causal effect using the frontdoor criterion:
P(Y|do(X)) = Σ_M P(M|X) * Σ_X' P(Y|X',M) * P(X')

This applies when a mediator M is observed that blocks all directed paths from X to Y,
and there is no unblocked backdoor path from X to M.

shape = [N_x, N_m, N_y, T]  (N_y is unused; reserved for API symmetry)
data[0] = X [T] — treatment (scalar per observation)
data[1] = M [T] — mediator (scalar per observation)
data[2] = Y [T] — outcome (scalar per observation)
Discretization uses equal-frequency binning on X and M (no optional bin array).
Returns causal_effect [N_x].
*/
type FrontdoorAdjustment struct{}

/*
NewFrontdoorAdjustment instantiates a FrontdoorAdjustment operation.
It implements Pearl's frontdoor criterion for causal identification.
*/
func NewFrontdoorAdjustment() *FrontdoorAdjustment {
	return &FrontdoorAdjustment{}
}

/*
Forward computes the frontdoor-adjusted causal effect.
*/
func (frontdoorAdjustment *FrontdoorAdjustment) Forward(shape []int, data ...[]float64) []float64 {
	if len(shape) < 4 {
		panic(fmt.Errorf("causal: FrontdoorAdjustment.Forward: len(shape)=%d, need >= 4", len(shape)))
	}

	if len(data) < 3 {
		panic(fmt.Errorf("causal: FrontdoorAdjustment.Forward: len(data)=%d, need >= 3", len(data)))
	}

	nx, nm, _, t := shape[0], shape[1], shape[2], shape[3]

	xVec := data[0]
	mVec := data[1]
	yVec := data[2]

	if len(xVec) < t || len(mVec) < t || len(yVec) < t {
		panic(fmt.Errorf(
			"causal: FrontdoorAdjustment.Forward: data lengths %d/%d/%d must all be >= T=%d",
			len(xVec), len(mVec), len(yVec), t,
		))
	}

	// Discretize X into nx bins and M into nm bins using equal-frequency binning.
	xBins := discretize(xVec[:t], nx)
	mBins := discretize(mVec[:t], nm)

	// P(X=x) — marginal distribution of X bins.
	pX := make([]float64, nx)

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		pX[xBins[obsIdx]]++
	}

	for binIdx := range pX {
		pX[binIdx] /= float64(t)
	}

	// P(M=m|X=x) — conditional distribution of M given X.
	pMGivenX := make([]float64, nm*nx)
	countX := make([]float64, nx)

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		xBin := xBins[obsIdx]
		mBin := mBins[obsIdx]
		pMGivenX[mBin*nx+xBin]++
		countX[xBin]++
	}

	for xBin := 0; xBin < nx; xBin++ {
		for mBin := 0; mBin < nm; mBin++ {
			if countX[xBin] > 0 {
				pMGivenX[mBin*nx+xBin] /= countX[xBin]
			}
		}
	}

	// E[Y|X'=x', M=m] — conditional expectation of Y given X' and M.
	eYGivenXM := make([]float64, nx*nm)
	countXM := make([]float64, nx*nm)

	for obsIdx := 0; obsIdx < t; obsIdx++ {
		xBin := xBins[obsIdx]
		mBin := mBins[obsIdx]
		eYGivenXM[xBin*nm+mBin] += yVec[obsIdx]
		countXM[xBin*nm+mBin]++
	}

	for xBin := 0; xBin < nx; xBin++ {
		for mBin := 0; mBin < nm; mBin++ {
			idxCell := xBin*nm + mBin

			if countXM[idxCell] > 0 {
				eYGivenXM[idxCell] /= countXM[idxCell]
				continue
			}

			eYGivenXM[idxCell] = math.NaN()
		}
	}

	// Frontdoor formula:
	// E[Y|do(X=x)] = Σ_m P(M=m|X=x) * Σ_x' E[Y|X'=x', M=m] * P(X'=x')
	causalEffect := make([]float64, nx)

	for xBin := 0; xBin < nx; xBin++ {
		effect := 0.0

		for mBin := 0; mBin < nm; mBin++ {
			// Compute Σ_x' E[Y|X'=x', M=m] * P(X'=x')
			innerSum := 0.0

			for xPrimeBin := 0; xPrimeBin < nx; xPrimeBin++ {
				cellMean := eYGivenXM[xPrimeBin*nm+mBin]

				if math.IsNaN(cellMean) {
					continue
				}

				innerSum += cellMean * pX[xPrimeBin]
			}

			effect += pMGivenX[mBin*nx+xBin] * innerSum
		}

		causalEffect[xBin] = effect
	}

	return causalEffect
}

/*
discretize assigns each value in data to one of nBins equal-frequency bins.
Returns a slice of bin indices [0, nBins).
*/
func discretize(data []float64, nBins int) []int {
	n := len(data)

	if nBins <= 0 || n == 0 {
		panic(fmt.Errorf("causal: discretize: need nBins > 0 and len(data) > 0 (got nBins=%d, n=%d)", nBins, n))
	}

	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)

	boundaries := make([]float64, nBins-1)

	for binIdx := 1; binIdx < nBins; binIdx++ {
		quantilePos := float64(binIdx) / float64(nBins) * float64(n)
		intPos := int(quantilePos)

		if intPos >= n {
			intPos = n - 1
		}

		boundaries[binIdx-1] = sorted[intPos]
	}

	bins := make([]int, n)

	for obsIdx, val := range data {
		bins[obsIdx] = sort.Search(len(boundaries), func(i int) bool { return val < boundaries[i] })
	}

	return bins
}
