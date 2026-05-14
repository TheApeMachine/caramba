//go:build arm64

package projection

func tiedEmbeddingKernel(dst, input, weight []float64, M, K, N int) {
	tiedEmbeddingMatmulNEON(dst, input, weight, M, K, N)
}

func tiedEmbeddingMatmulNEON(dst, input, weight []float64, M, K, N int) {
	projMatmulNEON(dst, input, weight, M, K, N)
}
