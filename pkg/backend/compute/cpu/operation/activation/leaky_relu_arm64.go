//go:build arm64

package activation

//go:noescape
func LeakyReLUNEON(dst, src []float64, alpha float64)

func applyLeakyReLU(dst, src []float64, alpha float64) {
	LeakyReLUNEON(dst, src, alpha)
}
