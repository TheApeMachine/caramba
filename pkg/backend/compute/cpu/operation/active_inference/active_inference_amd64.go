//go:build amd64

package active_inference

import (
	"fmt"
	"math"

	"golang.org/x/sys/cpu"
)

var useAVX2 bool
var useFMA bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
	useFMA = cpu.X86.HasFMA
}

//go:noescape
func freeEnergyAVX2(mu, logSigma []float64) float64

//go:noescape
func freeEnergySSE2(mu, logSigma []float64) float64

//go:noescape
func precisionWeightMulAVX2(dst, errVec, prec []float64)

//go:noescape
func precisionWeightMulSSE2(dst, errVec, prec []float64)

func applyFreeEnergy(mu, logSigma []float64, n int, expScratch []float64) float64 {
	// SIMD computes: sum(mu^2 + exp(logSigma) - logSigma - 1) * 0.5
	// exp is scalar so we build an exp buffer and then run SIMD on remainder terms.
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

	for idx := 0; idx < n; idx++ {
		expBuf[idx] = math.Exp(logSigmaUse[idx])
	}

	var sum float64

	if useAVX2 && useFMA {
		sum = freeEnergyAVX2(muUse, expBuf)
	} else {
		sum = freeEnergySSE2(muUse, expBuf)
	}

	for idx := 0; idx < n; idx++ {
		sum -= logSigmaUse[idx] + 1.0
	}

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

	// mu_new = mu - lr*(mu + pred_err)  → mu*(1-lr) - lr*pred_err
	// We use SIMD for the mu update (addVec + scale) and scalar loop for logSigma (requires exp).
	limit := alignedLen(n)

	if useAVX2 && useFMA {
		if limit > 0 {
			beliefUpdateMuAVX2(muOut[:limit], mu[:limit], predErr[:limit], lr)
		}
	} else if limit > 0 {
		beliefUpdateMuSSE2(muOut[:limit], mu[:limit], predErr[:limit], lr)
	}

	for idx := limit; idx < n; idx++ {
		muOut[idx] = mu[idx] - lr*(mu[idx]+predErr[idx])
	}

	for idx := 0; idx < n; idx++ {
		logSigOut[idx] = logSigma[idx] - lr*(math.Exp(logSigma[idx])-1.0)
	}
}

//go:noescape
func beliefUpdateMuAVX2(dst, mu, predErr []float64, lr float64)

//go:noescape
func beliefUpdateMuSSE2(dst, mu, predErr []float64, lr float64)

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

	for idx := 0; idx < fillCount; idx++ {
		prec[idx] = math.Exp(logPrec[idx])
	}

	for idx := fillCount; idx < n; idx++ {
		prec[idx] = 1.0
	}

	limit := alignedLen(n)

	if useAVX2 && useFMA {
		if limit > 0 {
			precisionWeightMulAVX2(dst[:limit], errVec[:limit], prec[:limit])
		}
	} else if limit > 0 {
		precisionWeightMulSSE2(dst[:limit], errVec[:limit], prec[:limit])
	}

	for idx := limit; idx < n; idx++ {
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

	for kIdx := 0; kIdx < k; kIdx++ {
		g := 0.0

		for iIdx := 0; iIdx < n; iIdx++ {
			q := qOutcomes[iIdx*k+kIdx]

			if q < 0 {
				q = 0
			}

			if q > 1 {
				q = 1
			}

			g -= q * math.Log(q+eps)
		}

		out[kIdx] = g
	}
}

func alignedLen(n int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return n / width * width
}
