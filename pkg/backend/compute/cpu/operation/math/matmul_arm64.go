//go:build arm64

package math

//go:noescape
func matmulNEON(dst, a, b []float64, M, K, N int)

func applyMatmul(dst, a, b []float64, M, K, N int) {
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var acc float64
			for k := 0; k < K; k++ {
				acc += a[i*K+k] * b[k*N+j]
			}
			dst[i*N+j] = acc
		}
	}
}
