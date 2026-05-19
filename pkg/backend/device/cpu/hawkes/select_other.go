//go:build !arm64

package hawkes

import "math"

func hawkesKernelMatrixScalar(
	eventTimes, out []float32,
	alpha, beta float32,
) {
	eventCount := len(eventTimes)

	for rowIndex := 0; rowIndex < eventCount; rowIndex++ {
		for colIndex := 0; colIndex < eventCount; colIndex++ {
			if colIndex >= rowIndex {
				out[rowIndex*eventCount+colIndex] = 0
				continue
			}

			delta := eventTimes[rowIndex] - eventTimes[colIndex]
			out[rowIndex*eventCount+colIndex] = alpha * float32(math.Exp(float64(-beta*delta)))
		}
	}
}

func HawkesIntensityNative(
	eventTimes, queryTimes, out []float32,
	mu, alpha, beta float32,
) {
	HawkesIntensityScalar(eventTimes, queryTimes, out, mu, alpha, beta)
}

func HawkesKernelMatrixNative(
	eventTimes, out []float32,
	alpha, beta float32,
) {
	hawkesKernelMatrixScalar(eventTimes, out, alpha, beta)
}

func HawkesLogLikelihoodNative(
	eventTimes []float32,
	totalT, mu, alpha, beta float32,
	out []float32,
) {
	HawkesLogLikelihoodScalar(eventTimes, totalT, mu, alpha, beta, out)
}

func MarkovMutualInformationNative(joint []float32, xCount, yCount int, out []float32) {
	MarkovMutualInformationScalar(joint, xCount, yCount, out)
}
