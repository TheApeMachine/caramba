//go:build amd64

package projection

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

//go:noescape
func projMatmulAVX2(dst, a, b []float64, M, K, N int)

//go:noescape
func projMatmulSSE2(dst, a, b []float64, M, K, N int)

func applyMatmul(dst, a, b []float64, M, K, N int) {
	if useAVX2 {
		projMatmulAVX2(dst, a, b, M, K, N)
	} else {
		projMatmulSSE2(dst, a, b, M, K, N)
	}
}
