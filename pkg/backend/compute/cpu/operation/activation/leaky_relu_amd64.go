//go:build amd64

package activation

//go:noescape
func LeakyReLUAVX2(dst, src []float64, alpha float64)

//go:noescape
func LeakyReLUSSE2(dst, src []float64, alpha float64)

func applyLeakyReLU(dst, src []float64, alpha float64) {
	if useAVX2 {
		LeakyReLUAVX2(dst, src, alpha)
	} else {
		LeakyReLUSSE2(dst, src, alpha)
	}
}
