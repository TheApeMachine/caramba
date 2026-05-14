//go:build !amd64 && !arm64

package shape

func reshapeKernel(dst, src []float64) {
	copy(dst, src)
}
