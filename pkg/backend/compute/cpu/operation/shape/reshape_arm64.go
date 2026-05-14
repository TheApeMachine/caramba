//go:build arm64

package shape

func reshapeKernel(dst, src []float64) {
	CopyNEON(dst, src)
}
