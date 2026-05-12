//go:build arm64

package active_inference

import (
	"fmt"

	mathops "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
)

//go:noescape
func freeEnergyNEON(mu, expSigma []float64) float64

//go:noescape
func precisionWeightMulNEON(dst, errVec, prec []float64)

//go:noescape
func beliefUpdateMuNEON(dst, mu, predErr []float64, lr float64)

func applyFreeEnergy(mu, logSigma []float64, n int, expScratch []float64) float64 {
	if n < 0 {
		panic("active_inference: applyFreeEnergy: n < 0")
	}

	if len(mu) < n || len(logSigma) < n {
		panic(fmt.Sprintf(
			"active_inference: applyFreeEnergy: need len(mu) >= n and len(logSigma) >= n (got %d, %d, n=%d)",
			len(mu), len(logSigma), n,
		))
	}

	muUse := mu[:n]
	logSigmaUse := logSigma[:n]

	var expBuf []float64

	if expScratch != nil {
		if len(expScratch) < n {
			panic(fmt.Sprintf("active_inference: applyFreeEnergy: expScratch len=%d < n=%d", len(expScratch), n))
		}

		expBuf = expScratch[:n]
	} else {
		expBuf = make([]float64, n)
	}

	mathops.ExpVec(expBuf, logSigmaUse)

	sum := freeEnergyNEON(muUse, expBuf)
	sum -= mathops.ReduceSum(logSigmaUse) + float64(n)

	return 0.5 * sum
}

func applyBeliefUpdate(muOut, logSigOut, mu, logSigma, predErr []float64, lr float64, n int) {
	if n < 0 {
		panic("active_inference: applyBeliefUpdate: n < 0")
	}

	if len(muOut) < n || len(logSigOut) < n || len(mu) < n || len(logSigma) < n || len(predErr) < n {
		panic(fmt.Sprintf(
			"active_inference: applyBeliefUpdate: slice shorter than n=%d (muOut=%d logSigOut=%d mu=%d logSigma=%d predErr=%d)",
			n, len(muOut), len(logSigOut), len(mu), len(logSigma), len(predErr),
		))
	}

	pairLimit := n - (n % 2)

	if pairLimit > 0 {
		beliefUpdateMuNEON(muOut[:pairLimit], mu[:pairLimit], predErr[:pairLimit], lr)
	}

	for idx := pairLimit; idx < n; idx++ {
		muOut[idx] = mu[idx] - lr*(mu[idx]+predErr[idx])
	}

	expBuf := make([]float64, n)
	mathops.ExpVec(expBuf, logSigma[:n])
	mathops.AddScalarVec(expBuf, -1.0)
	copy(logSigOut[:n], logSigma[:n])
	mathops.AddScaledVec(logSigOut[:n], expBuf, -lr)
}

func applyPrecisionWeight(dst, errVec, logPrec []float64, n int, precScratch []float64) {
	if n < 0 {
		panic("active_inference: applyPrecisionWeight: n < 0")
	}

	if len(dst) < n || len(errVec) < n {
		panic(fmt.Sprintf(
			"active_inference: applyPrecisionWeight: need len(dst) >= n and len(errVec) >= n (got %d, %d, n=%d)",
			len(dst), len(errVec), n,
		))
	}

	var prec []float64

	if precScratch != nil {
		if len(precScratch) < n {
			panic(fmt.Sprintf("active_inference: applyPrecisionWeight: precScratch len=%d < n=%d", len(precScratch), n))
		}

		prec = precScratch[:n]
	} else {
		prec = make([]float64, n)
	}

	fillCount := min(n, len(logPrec))

	if fillCount > 0 {
		mathops.ExpVec(prec[:fillCount], logPrec[:fillCount])
	}

	for idx := fillCount; idx < n; idx++ {
		prec[idx] = 1.0
	}

	pairLimit := n - (n % 2)

	if pairLimit > 0 {
		precisionWeightMulNEON(dst[:pairLimit], errVec[:pairLimit], prec[:pairLimit])
	}

	for idx := pairLimit; idx < n; idx++ {
		dst[idx] = errVec[idx] * prec[idx]
	}
}

func applyExpectedFreeEnergy(out, qOutcomes []float64, n, k int) {
	if n <= 0 || k <= 0 {
		panic(fmt.Sprintf("active_inference: applyExpectedFreeEnergy: need n > 0 and k > 0 (got n=%d k=%d)", n, k))
	}

	need := n * k

	if len(qOutcomes) < need {
		panic(fmt.Sprintf(
			"active_inference: applyExpectedFreeEnergy: len(qOutcomes)=%d < n*k=%d (row-major q[i,k]=qOutcomes[i*k+k])",
			len(qOutcomes), need,
		))
	}

	if len(out) < k {
		panic(fmt.Sprintf("active_inference: applyExpectedFreeEnergy: len(out)=%d < k=%d", len(out), k))
	}

	const eps = 1e-12

	col := make([]float64, n)
	logBuf := make([]float64, n)
	prod := make([]float64, n)

	for kIdx := 0; kIdx < k; kIdx++ {
		for iIdx := 0; iIdx < n; iIdx++ {
			col[iIdx] = qOutcomes[iIdx*k+kIdx]
		}

		mathops.ClampVec(col, 0, 1)
		copy(logBuf, col)
		mathops.AddScalarVec(logBuf, eps)
		mathops.LogVec(logBuf, logBuf)
		mathops.MulVec(prod, col, logBuf)
		out[kIdx] = -mathops.ReduceSum(prod)
	}
}
