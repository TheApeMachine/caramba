//go:build !amd64 && !arm64

package projection

func tiedEmbeddingKernel(dst, input, weight []float64, M, K, N int) {
	projectionMatmulGeneric(dst, input, weight, M, K, N)
}
