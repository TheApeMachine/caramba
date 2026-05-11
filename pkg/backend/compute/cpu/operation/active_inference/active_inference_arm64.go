//go:build arm64

package active_inference

import "math"

//go:noescape
func freeEnergyNEON(mu, expSigma []float64) float64

//go:noescape
func precisionWeightMulNEON(dst, errVec, prec []float64)

//go:noescape
func beliefUpdateMuNEON(dst, mu, predErr []float64, lr float64)

func applyFreeEnergy(mu, logSigma []float64, n int) float64 {
	expBuf := make([]float64, n)

	for idx := range logSigma {
		expBuf[idx] = math.Exp(logSigma[idx])
	}

	sum := freeEnergyNEON(mu, expBuf)

	for idx := range logSigma {
		sum -= logSigma[idx] + 1.0
	}

	return 0.5 * sum
}

func applyBeliefUpdate(muOut, logSigOut, mu, logSigma, predErr []float64, lr float64, n int) {
	beliefUpdateMuNEON(muOut, mu, predErr, lr)

	if n%2 != 0 {
		last := n - 1
		muOut[last] = mu[last] - lr*(mu[last]+predErr[last])
	}

	for idx := range logSigma {
		logSigOut[idx] = logSigma[idx] - lr*(math.Exp(logSigma[idx])-1.0)
	}
}

func applyPrecisionWeight(dst, errVec, logPrec []float64, n int) {
	prec := make([]float64, n)

	for idx := range logPrec {
		prec[idx] = math.Exp(logPrec[idx])
	}

	precisionWeightMulNEON(dst, errVec, prec)

	if n%2 != 0 {
		last := n - 1
		dst[last] = errVec[last] * prec[last]
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
