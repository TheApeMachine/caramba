//go:build arm64

package activation

//go:noescape
func SigmoidNEON(dst, src []float64)

func applySigmoid(dst, src []float64) {
	SigmoidNEON(dst, src)
}
