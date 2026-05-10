//go:build amd64

package shape

import "golang.org/x/sys/cpu"

var useAVX2 bool

func init() {
	useAVX2 = cpu.X86.HasAVX2
}

// CopyAVX2 copies src into dst using 256-bit AVX2 stores (4 float64s/iter).
//
//go:noescape
func CopyAVX2(dst, src []float64)

// CopySSE2 copies src into dst using 128-bit SSE2 stores (2 float64s/iter).
//
//go:noescape
func CopySSE2(dst, src []float64)

// copyBlock copies src into dst using the best available SIMD path.
func copyBlock(dst, src []float64) {
	if useAVX2 {
		CopyAVX2(dst, src)
	} else {
		CopySSE2(dst, src)
	}
}
