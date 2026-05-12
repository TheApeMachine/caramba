//go:build arm64

package hawkes

import "math"

//go:noescape
func expSumNEON(expBuf []float64) float64

func applyIntensity(out, times, alpha, beta, mu []float64, t float64, K, T int) {
	cutoff := 0
	for cutoff < T && times[cutoff] < t {
		cutoff++
	}

	validTimes := times[:cutoff]
	n := len(validTimes)

	if n == 0 {
		for k := 0; k < K; k++ {
			out[k] = mu[k]
		}
		return
	}

	expBuf := make([]float64, n)

	for k := 0; k < K; k++ {
		bk := beta[k]
		for i := 0; i < n; i++ {
			expBuf[i] = math.Exp(-bk * (t - validTimes[i]))
		}

		out[k] = mu[k] + alpha[k]*expSumNEON(expBuf)
	}
}

// applyKernelMatrix fills the upper-triangular excitation kernel. times must be non-decreasing.
func applyKernelMatrix(out, times []float64, alpha, beta float64, T int) {
	if T <= 0 || len(times) < T || len(out) < T*T {
		panic("hawkes: applyKernelMatrix: need T > 0, len(times) >= T, len(out) >= T*T")
	}

	for row := 0; row < T; row++ {
		if row > 0 && times[row] < times[row-1] {
			panic("hawkes: applyKernelMatrix: times must be non-decreasing")
		}
	}

	exponents := make([]float64, T)

	for row := 0; row < T; row++ {
		rowLen := T - row - 1

		if rowLen == 0 {
			continue
		}

		applyKernelMatrixRowNEON(
			out[row*T+row+1:row*T+T],
			exponents[:rowLen],
			times[row+1:T],
			alpha,
			beta,
			times[row],
		)
	}
}

func applyKernelMatrixRowNEON(out, exponents, times []float64, alpha, beta, origin float64) {
	for index, eventTime := range times {
		exponents[index] = math.Exp(-beta * (eventTime - origin))
	}

	for index, exponent := range exponents {
		out[index] = alpha * exponent
	}
}
