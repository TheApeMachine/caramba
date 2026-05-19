//go:build !arm64

package cpu

func HawkesIntensityNative(
	eventTimes, queryTimes, out []float32,
	mu, alpha, beta float32,
) {
	hawkesIntensityScalar(eventTimes, queryTimes, out, mu, alpha, beta)
}

func HawkesKernelMatrixNative(
	eventTimes, out []float32,
	alpha, beta float32,
) {
	hawkesKernelMatrixScalar(eventTimes, out, alpha, beta)
}
