//go:build amd64

package projection

func linearKernel(dst, input, weight, bias []float64, M, K, N int) {
	if useAVX2 && useFMA {
		linearMatmulAVX2(dst, input, weight, M, K, N)
	} else {
		linearMatmulSSE2(dst, input, weight, M, K, N)
	}

	if len(bias) != 0 {
		addBias(dst, bias, M, N)
	}
}

// linearMatmulAVX2 is the AVX2-backed matmul for the Linear op.
func linearMatmulAVX2(dst, input, weight []float64, M, K, N int) {
	projMatmulAVX2(dst, input, weight, M, K, N)
}

// linearMatmulSSE2 is the SSE2 fallback for the Linear op.
func linearMatmulSSE2(dst, input, weight []float64, M, K, N int) {
	projMatmulSSE2(dst, input, weight, M, K, N)
}
