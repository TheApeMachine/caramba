//go:build amd64

package projection

// fusedQKVMatmulAVX2 delegates to the shared projection matmul primitive.
func fusedQKVMatmulAVX2(dst, a, b []float64, M, K, N int) {
	projMatmulAVX2(dst, a, b, M, K, N)
}

// fusedQKVMatmulSSE2 is the SSE2 fallback.
func fusedQKVMatmulSSE2(dst, a, b []float64, M, K, N int) {
	projMatmulSSE2(dst, a, b, M, K, N)
}
