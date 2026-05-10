//go:build amd64

package activation

//go:noescape
func ReLUAVX2(dst, src []float64)

//go:noescape
func ReLUSSE2(dst, src []float64)

func applyReLU(dst, src []float64) {
	if useAVX2 {
		ReLUAVX2(dst, src)
	} else {
		ReLUSSE2(dst, src)
	}
}
