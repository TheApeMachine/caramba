//go:build arm64

package shape

// CopyNEON copies src into dst using ARM NEON 64-bit SIMD (2 float64s/iter).
//
//go:noescape
func CopyNEON(dst, src []float64)

// copyBlock copies src into dst using the NEON SIMD path.
func copyBlock(dst, src []float64) {
	CopyNEON(dst, src)
}
