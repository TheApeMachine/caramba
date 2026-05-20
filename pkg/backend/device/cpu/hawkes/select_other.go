//go:build !arm64 && !amd64

package hawkes

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
