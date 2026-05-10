//go:build arm64

package activation

//go:noescape
func ReLUNEON(dst, src []float64)

func applyReLU(dst, src []float64) {
	ReLUNEON(dst, src)
}
