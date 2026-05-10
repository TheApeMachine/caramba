//go:build amd64

package projection

// tiedEmbMatmulAVX2 delegates to the shared projection matmul primitive.
func tiedEmbMatmulAVX2(dst, a, b []float64, M, K, N int) {
	projMatmulAVX2(dst, a, b, M, K, N)
}

// tiedEmbMatmulSSE2 is the SSE2 fallback.
func tiedEmbMatmulSSE2(dst, a, b []float64, M, K, N int) {
	projMatmulSSE2(dst, a, b, M, K, N)
}
