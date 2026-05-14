//go:build arm64

package math

//go:noescape
func matmulNEON(dst, a, b []float64, M, K, N int)

func matmulKernel(dst, a, b []float64, M, K, N int) {
	matmulNEON(dst, a, b, M, K, N)
}
