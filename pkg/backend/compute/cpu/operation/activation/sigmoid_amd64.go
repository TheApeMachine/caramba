//go:build amd64

package activation

//go:noescape
func SigmoidAVX2(dst, src []float64)

//go:noescape
func SigmoidSSE2(dst, src []float64)

func applySigmoid(dst, src []float64) {
	if useAVX2 {
		SigmoidAVX2(dst, src)
	} else {
		SigmoidSSE2(dst, src)
	}
}
