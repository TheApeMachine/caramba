//go:build arm64

package hawkes

import stdmath "math"

//go:noescape
func expSumNEON(times []float64, t, beta float64) float64

func applyIntensity(out, times, alpha, beta, mu []float64, t float64, K, T int) {
	applyIntensityScalar(out, times, alpha, beta, mu, t, K, T)
}

func applyKernelMatrix(out, times []float64, alpha, beta float64, T int) {
	for row := range T {
		ti := times[row]

		for col := row + 1; col < T; col++ {
			out[row*T+col] = alpha * stdmath.Exp(-beta*(times[col]-ti))
		}
	}
}

func applyLogLikelihoodSumLog(intensities []float64, T int) float64 {
	return applyLogLikelihoodScalar(intensities, T)
}
