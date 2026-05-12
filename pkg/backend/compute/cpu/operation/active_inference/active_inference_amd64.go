//go:build amd64

package active_inference

import (
	"fmt"

	"golang.org/x/sys/cpu"

	mathops "github.com/theapemachine/caramba/pkg/backend/compute/cpu/operation/math"
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

//go:noescape
func beliefUpdateMuAVX2(dst, mu, predErr []float64, lr float64)

//go:noescape
func beliefUpdateMuSSE2(dst, mu, predErr []float64, lr float64)

// applyFreeEnergy: F = 0.5 * sum(mu^2 + exp(logSigma) - logSigma - 1)
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

	var sum float64

	if useAVX2 && useFMA {
		sum = freeEnergyAVX2(muUse, expBuf)
	} else {
		sum = freeEnergySSE2(muUse, expBuf)
	}

	sum -= mathops.ReduceSum(logSigmaUse) + float64(n)

	return 0.5 * sum
}

// applyBeliefUpdate:
//   muOut[i]      = mu[i] - lr * (mu[i] + predErr[i])
//   logSigOut[i]  = logSigma[i] - lr * (exp(logSigma[i]) - 1)
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

	expBuf := make([]float64, n)
	mathops.ExpVec(expBuf, logSigma[:n])
	mathops.AddScalarVec(expBuf, -1.0)
	copy(logSigOut[:n], logSigma[:n])
	mathops.AddScaledVec(logSigOut[:n], expBuf, -lr)
}

// applyPrecisionWeight: dst[i] = errVec[i] * exp(logPrec[i])
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

// applyExpectedFreeEnergy: out[k] = -sum_i q_i,k * log(q_i,k + eps), with q clamped to [0,1].
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

func alignedLen(n int) int {
	width := 2

	if useAVX2 {
		width = 4
	}

	return n / width * width
}
