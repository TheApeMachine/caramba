//go:build arm64

package activation

//go:noescape
func GeLUNEON(dst, x []float64)

func geluKernel(dst, src []float64) {
	GeLUNEON(dst, src)
}
