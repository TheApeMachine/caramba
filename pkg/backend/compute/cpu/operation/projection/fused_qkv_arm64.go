//go:build arm64

package projection

func fusedQKVKernel(dst, input, weight, bias []float64, M, K, N int) {
	fusedQKVMatmulNEON(dst, input, weight, M, K, N)

	if len(bias) != 0 {
		addBias(dst, bias, M, N)
	}
}

func fusedQKVMatmulNEON(dst, input, weight []float64, M, K, N int) {
	projMatmulNEON(dst, input, weight, M, K, N)
}
