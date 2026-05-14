//go:build amd64

package math

//go:noescape
func matmulAVX2(dst, a, b []float64, M, K, N int)

//go:noescape
func matmulSSE2(dst, a, b []float64, M, K, N int)

func matmulKernel(dst, a, b []float64, M, K, N int) {
	if useAVX2 && useFMA {
		matmulAVX2(dst, a, b, M, K, N)
	} else {
		matmulSSE2(dst, a, b, M, K, N)
	}
}
