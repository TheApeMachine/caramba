//go:build !arm64

package cpu

func HawkesLogLikelihoodNative(
	eventTimes []float32,
	totalT, mu, alpha, beta float32,
	out []float32,
) {
	hawkesLogLikelihoodScalar(eventTimes, totalT, mu, alpha, beta, out)
}

func MarkovMutualInformationNative(joint []float32, xCount, yCount int, out []float32) {
	markovMutualInformationScalar(joint, xCount, yCount, out)
}
