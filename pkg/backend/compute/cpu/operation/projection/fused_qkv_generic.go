//go:build !amd64 && !arm64

package projection

func fusedQKVKernel(dst, input, weight, bias []float64, M, K, N int) {
	projectionMatmulGeneric(dst, input, weight, M, K, N)

	if len(bias) != 0 {
		addBias(dst, bias, M, N)
	}
}
