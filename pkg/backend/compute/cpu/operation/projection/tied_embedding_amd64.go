//go:build amd64

package projection

func tiedEmbeddingKernel(dst, input, weight []float64, M, K, N int) {
	if useAVX2 && useFMA {
		tiedEmbeddingMatmulAVX2(dst, input, weight, M, K, N)
		return
	}

	tiedEmbeddingMatmulSSE2(dst, input, weight, M, K, N)
}

// tiedEmbeddingMatmulAVX2 delegates to the shared projection matmul primitive.
func tiedEmbeddingMatmulAVX2(dst, input, weight []float64, M, K, N int) {
	projMatmulAVX2(dst, input, weight, M, K, N)
}

// tiedEmbeddingMatmulSSE2 is the SSE2 fallback.
func tiedEmbeddingMatmulSSE2(dst, input, weight []float64, M, K, N int) {
	projMatmulSSE2(dst, input, weight, M, K, N)
}
