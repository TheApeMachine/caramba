//go:build arm64

package projection

func linearKernel(dst, input, weight, bias []float64, M, K, N int) {
	linearMatmulNEON(dst, input, weight, M, K, N)

	if len(bias) != 0 {
		addBias(dst, bias, M, N)
	}
}

// linearMatmulNEON is the NEON-backed matmul for the Linear op.
func linearMatmulNEON(dst, input, weight []float64, M, K, N int) {
	projMatmulNEON(dst, input, weight, M, K, N)
}
