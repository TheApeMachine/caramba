//go:build amd64

package projection

// linearMatmulAVX2 is the AVX2-backed matmul for the Linear op.
// Dispatched via applyMatmul (primitives_amd64.go).
func linearMatmulAVX2(dst, a, b []float64, M, K, N int) {
	projMatmulAVX2(dst, a, b, M, K, N)
}

// linearMatmulSSE2 is the SSE2 fallback for the Linear op.
func linearMatmulSSE2(dst, a, b []float64, M, K, N int) {
	projMatmulSSE2(dst, a, b, M, K, N)
}
