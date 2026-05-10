//go:build arm64

package projection

// tiedEmbMatmulNEON delegates to the shared projection NEON matmul primitive.
func tiedEmbMatmulNEON(dst, a, b []float64, M, K, N int) {
	projMatmulNEON(dst, a, b, M, K, N)
}
