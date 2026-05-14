//go:build amd64

package projection

func fusedQKVKernel(dst, input, weight, bias []float64, M, K, N int) {
	if useAVX2 && useFMA {
		fusedQKVMatmulAVX2(dst, input, weight, M, K, N)
	} else {
		fusedQKVMatmulSSE2(dst, input, weight, M, K, N)
	}

	if len(bias) != 0 {
		addBias(dst, bias, M, N)
	}
}

// fusedQKVMatmulAVX2 delegates to the shared projection matmul primitive.
func fusedQKVMatmulAVX2(dst, input, weight []float64, M, K, N int) {
	projMatmulAVX2(dst, input, weight, M, K, N)
}

// fusedQKVMatmulSSE2 is the SSE2 fallback.
func fusedQKVMatmulSSE2(dst, input, weight []float64, M, K, N int) {
	projMatmulSSE2(dst, input, weight, M, K, N)
}
