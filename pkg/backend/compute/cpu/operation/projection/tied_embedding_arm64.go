//go:build arm64

package projection

//go:noescape
func tiedEmbeddingMatmulNEON(dst, input, weight []float64, M, K, N int)

func tiedEmbeddingKernel(dst, input, weight []float64, M, K, N int) {
	tiedEmbeddingMatmulNEON(dst, input, weight, M, K, N)
}
