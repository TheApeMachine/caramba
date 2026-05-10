//go:build arm64

package projection

//go:noescape
func projMatmulNEON(dst, a, b []float64, M, K, N int)

func applyMatmul(dst, a, b []float64, M, K, N int) {
	projMatmulNEON(dst, a, b, M, K, N)
}
