//go:build arm64

package projection

// fusedQKVMatmulNEON delegates to the shared projection NEON matmul primitive.
func fusedQKVMatmulNEON(dst, a, b []float64, M, K, N int) {
	projMatmulNEON(dst, a, b, M, K, N)
}
