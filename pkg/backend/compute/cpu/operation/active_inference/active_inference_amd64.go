//go:build amd64

package active_inference

import (
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

func applyFreeEnergy(mu, logSigma []float64, n int) float64 {
	// SIMD computes: sum(mu^2 + exp(logSigma) - logSigma - 1) * 0.5
	// exp is scalar so we build an exp buffer and then run SIMD on remainder terms.
	expBuf := make([]float64, n)

	for idx := range logSigma {
		expBuf[idx] = math.Exp(logSigma[idx])
	}

	var sum float64

	if useAVX2 && useFMA {
		sum = freeEnergyAVX2(mu, expBuf)
	} else {
		sum = freeEnergySSE2(mu, expBuf)
	}

	// Subtract logSigma sum and constant term — scalar (log is cheap here)
	for idx := range logSigma {
		sum -= logSigma[idx] + 1.0
	}

	return 0.5 * sum
}

func applyBeliefUpdate(muOut, logSigOut, mu, logSigma, predErr []float64, lr float64, n int) {
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

	for idx := range logSigma {
		logSigOut[idx] = logSigma[idx] - lr*(math.Exp(logSigma[idx])-1.0)
	}
}

//go:noescape
func beliefUpdateMuAVX2(dst, mu, predErr []float64, lr float64)

//go:noescape
func beliefUpdateMuSSE2(dst, mu, predErr []float64, lr float64)

func applyPrecisionWeight(dst, errVec, logPrec []float64, n int) {
	prec := make([]float64, n)

	for idx := range logPrec {
		prec[idx] = math.Exp(logPrec[idx])
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
	const eps = 1e-12

	for kIdx := 0; kIdx < k; kIdx++ {
		g := 0.0

		for iIdx := 0; iIdx < n; iIdx++ {
			q := qOutcomes[iIdx*k+kIdx]
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
