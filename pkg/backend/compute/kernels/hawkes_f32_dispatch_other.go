//go:build !arm64

package kernels

func hawkesIntensityNative(
	eventTimes, queryTimes, out []float32,
	mu, alpha, beta float32,
) {
	hawkesIntensityScalar(eventTimes, queryTimes, out, mu, alpha, beta)
}

func hawkesKernelMatrixNative(
	eventTimes, out []float32,
	alpha, beta float32,
) {
	hawkesKernelMatrixScalar(eventTimes, out, alpha, beta)
}
