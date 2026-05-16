//go:build amd64

package projection

//go:noescape
func tiedEmbeddingMatmulAVX2(dst, input, weight []float64, M, K, N int)

//go:noescape
func tiedEmbeddingMatmulSSE2(dst, input, weight []float64, M, K, N int)

func tiedEmbeddingKernel(dst, input, weight []float64, M, K, N int) {
	if useAVX2 && useFMA {
		tiedEmbeddingMatmulAVX2(dst, input, weight, M, K, N)
		return
	}

	tiedEmbeddingMatmulSSE2(dst, input, weight, M, K, N)
}
