package causal

import "fmt"

/*
FrontdoorAdjustment computes the causal effect using the frontdoor criterion:
P(Y|do(X)) = Σ_M P(M|X) * Σ_X' P(Y|X',M) * P(X')

This applies when a mediator M is observed that blocks all directed paths from X to Y,
and there is no unblocked backdoor path from X to M.

shape = [N_x, N_m, N_y, T]
data[0] = X [T] — treatment (scalar per observation)
data[1] = M [T] — mediator (scalar per observation)
data[2] = Y [T] — outcome (scalar per observation)
data[3] = X_bins [N_x] — bin boundaries for discretizing X
Returns causal_effect [N_x] — causal effect estimate per X bin.
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
		panic(fmt.Errorf("causal: FrontdoorAdjustment.Forward: len(shape)=%d, need >= 4", len(shape)).Error())
	}

	if len(data) < 3 {
		panic(fmt.Errorf("causal: FrontdoorAdjustment.Forward: len(data)=%d, need >= 3", len(data)).Error())
	}

	nx, nm, _, t := shape[0], shape[1], shape[2], shape[3]

	xVec := data[0]
	mVec := data[1]
	yVec := data[2]

	if len(xVec) < t || len(mVec) < t || len(yVec) < t {
		panic(fmt.Errorf(
			"causal: FrontdoorAdjustment.Forward: data lengths %d/%d/%d must all be >= T=%d",
			len(xVec), len(mVec), len(yVec), t,
		).Error())
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
			if countXM[xBin*nm+mBin] > 0 {
				eYGivenXM[xBin*nm+mBin] /= countXM[xBin*nm+mBin]
			}
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
				innerSum += eYGivenXM[xPrimeBin*nm+mBin] * pX[xPrimeBin]
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
		return make([]int, n)
	}

	// Sort copy for quantile computation.
	sorted := make([]float64, n)
	copy(sorted, data)
	sortFloat64(sorted)

	// Compute bin boundaries at equal quantile points.
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
		bin := 0

		for boundaryIdx, boundary := range boundaries {
			if val >= boundary {
				bin = boundaryIdx + 1
			}
		}

		bins[obsIdx] = bin
	}

	return bins
}

/*
sortFloat64 sorts a float64 slice in-place using insertion sort for small slices
and a recursive quicksort for larger ones.
*/
func sortFloat64(a []float64) {
	if len(a) <= 1 {
		return
	}

	if len(a) <= 16 {
		for idx := 1; idx < len(a); idx++ {
			key := a[idx]
			j := idx - 1

			for j >= 0 && a[j] > key {
				a[j+1] = a[j]
				j--
			}

			a[j+1] = key
		}

		return
	}

	pivot := a[len(a)/2]
	left := 0
	right := len(a) - 1

	for left <= right {
		for a[left] < pivot {
			left++
		}

		for a[right] > pivot {
			right--
		}

		if left <= right {
			a[left], a[right] = a[right], a[left]
			left++
			right--
		}
	}

	sortFloat64(a[:right+1])
	sortFloat64(a[left:])
}
