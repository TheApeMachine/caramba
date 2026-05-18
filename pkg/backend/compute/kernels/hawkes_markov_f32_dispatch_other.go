//go:build !arm64

package kernels

func hawkesLogLikelihoodNative(
	eventTimes []float32,
	totalT, mu, alpha, beta float32,
	out []float32,
) {
	hawkesLogLikelihoodScalar(eventTimes, totalT, mu, alpha, beta, out)
}

func markovMutualInformationNative(joint []float32, xCount, yCount int, out []float32) {
	markovMutualInformationScalar(joint, xCount, yCount, out)
}
