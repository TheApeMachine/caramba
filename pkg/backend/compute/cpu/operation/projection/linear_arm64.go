//go:build arm64

package projection

// linearMatmulNEON is the NEON-backed matmul for the Linear op.
func linearMatmulNEON(dst, a, b []float64, M, K, N int) {
	projMatmulNEON(dst, a, b, M, K, N)
}
