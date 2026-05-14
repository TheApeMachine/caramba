package causal

import (
	"fmt"
	"math"
	"sort"

	"github.com/theapemachine/caramba/pkg/backend/compute/state"
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
func (frontdoorAdjustment *FrontdoorAdjustment) Forward(
	stateDict *state.Dict,
) (*state.Dict, error) {
	shape := stateDict.OperationShape()

	if len(shape) < 4 {
		return nil, fmt.Errorf("causal.frontdoor_adjustment: len(shape)=%d, need >= 4", len(shape))
	}

	if err := stateDict.RequireOperationInputs("causal.frontdoor_adjustment", 3); err != nil {
		return nil, err
	}

	treatmentBins := shape[0]
	mediatorBins := shape[1]
	samples := shape[3]

	xVec := stateDict.Inputs[0]
	mVec := stateDict.Inputs[1]
	yVec := stateDict.Inputs[2]

	if treatmentBins <= 0 || mediatorBins <= 0 || samples <= 0 {
		return nil, fmt.Errorf(
			"causal.frontdoor_adjustment: need N_x > 0, N_m > 0, T > 0 (got N_x=%d N_m=%d T=%d)",
			treatmentBins, mediatorBins, samples,
		)
	}

	if len(xVec) < samples || len(mVec) < samples || len(yVec) < samples {
		return nil, fmt.Errorf(
			"causal.frontdoor_adjustment: data lengths %d/%d/%d must all be >= T=%d",
			len(xVec), len(mVec), len(yVec), samples,
		)
	}

	// Discretize X into nx bins and M into nm bins using equal-frequency binning.
	xBins := discretize(xVec[:samples], treatmentBins)
	mBins := discretize(mVec[:samples], mediatorBins)

	// P(X=x) — marginal distribution of X bins.
	pX := make([]float64, treatmentBins)

	for obsIdx := 0; obsIdx < samples; obsIdx++ {
		pX[xBins[obsIdx]]++
	}

	for binIdx := range pX {
		pX[binIdx] /= float64(samples)
	}

	// P(M=m|X=x) — conditional distribution of M given X.
	pMGivenX := make([]float64, mediatorBins*treatmentBins)
	countX := make([]float64, treatmentBins)

	for obsIdx := 0; obsIdx < samples; obsIdx++ {
		xBin := xBins[obsIdx]
		mBin := mBins[obsIdx]
		pMGivenX[mBin*treatmentBins+xBin]++
		countX[xBin]++
	}

	for xBin := 0; xBin < treatmentBins; xBin++ {
		for mBin := 0; mBin < mediatorBins; mBin++ {
			if countX[xBin] > 0 {
				pMGivenX[mBin*treatmentBins+xBin] /= countX[xBin]
			}
		}
	}

	// E[Y|X'=x', M=m] — conditional expectation of Y given X' and M.
	eYGivenXM := make([]float64, treatmentBins*mediatorBins)
	countXM := make([]float64, treatmentBins*mediatorBins)

	for obsIdx := 0; obsIdx < samples; obsIdx++ {
		xBin := xBins[obsIdx]
		mBin := mBins[obsIdx]
		eYGivenXM[xBin*mediatorBins+mBin] += yVec[obsIdx]
		countXM[xBin*mediatorBins+mBin]++
	}

	for xBin := 0; xBin < treatmentBins; xBin++ {
		for mBin := 0; mBin < mediatorBins; mBin++ {
			idxCell := xBin*mediatorBins + mBin

			if countXM[idxCell] > 0 {
				eYGivenXM[idxCell] /= countXM[idxCell]
				continue
			}

			eYGivenXM[idxCell] = math.NaN()
		}
	}

	// Frontdoor formula:
	// E[Y|do(X=x)] = Σ_m P(M=m|X=x) * Σ_x' E[Y|X'=x', M=m] * P(X'=x')
	causalEffect := make([]float64, treatmentBins)

	for xBin := 0; xBin < treatmentBins; xBin++ {
		effect := 0.0

		for mBin := 0; mBin < mediatorBins; mBin++ {
			// Compute Σ_x' E[Y|X'=x', M=m] * P(X'=x')
			innerSum := 0.0

			for xPrimeBin := 0; xPrimeBin < treatmentBins; xPrimeBin++ {
				cellMean := eYGivenXM[xPrimeBin*mediatorBins+mBin]

				if math.IsNaN(cellMean) {
					continue
				}

				innerSum += cellMean * pX[xPrimeBin]
			}

			effect += pMGivenX[mBin*treatmentBins+xBin] * innerSum
		}

		causalEffect[xBin] = effect
	}

	stateDict.SetOperationOutput(causalEffect)

	return stateDict, nil
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
