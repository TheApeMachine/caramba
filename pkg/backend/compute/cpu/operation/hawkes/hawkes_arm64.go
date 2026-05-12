//go:build arm64

package hawkes

import "math"

// expSumNEON is defined in hawkes_neon_arm64.s; not yet wired into applyIntensity (scalar path only).
//
//go:noescape
func expSumNEON(expBuf []float64) float64

// applyIntensity delegates to applyIntensityScalar. TODO: replace with a NEON vectorized
// intensity routine (see applyIntensityScalar) when a correct asm kernel is ready.
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

// applyKernelMatrix fills the upper-triangular excitation kernel. times must be non-decreasing;
// callers (e.g. Hawkes Forward) should sort event times before invoking. TODO: optional NEON
// exp for the inner loop once numerical parity with scalar Exp is validated.
func applyKernelMatrix(out, times []float64, alpha, beta float64, T int) {
	if T <= 0 || len(times) < T || len(out) < T*T {
		panic("hawkes: applyKernelMatrix: need T > 0, len(times) >= T, len(out) >= T*T")
	}

	for row := 0; row < T; row++ {
		if row > 0 && times[row] < times[row-1] {
			panic("hawkes: applyKernelMatrix: times must be non-decreasing")
		}
	}

	for row := 0; row < T; row++ {
		ti := times[row]

		for col := row + 1; col < T; col++ {
			out[row*T+col] = alpha * math.Exp(-beta*(times[col]-ti))
		}
	}
}
