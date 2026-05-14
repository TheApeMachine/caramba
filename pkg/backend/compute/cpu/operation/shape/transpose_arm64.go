//go:build arm64

package shape

// CopyNEON copies src into dst using ARM NEON 64-bit SIMD (2 float64s/iter).
//
//go:noescape
func CopyNEON(dst, src []float64)

func copyBlock(dst, src []float64) {
	CopyNEON(dst, src)
}

func transposeKernel(dst, src []float64, shape []int, dim0, dim1 int) {
	transposeGenericKernel(dst, src, shape, dim0, dim1)
}
