//go:build arm64

package activation

//go:noescape
func SigmoidNEON(dst, src []float64)

func applySigmoid(dst, src []float64) {
	limit := len(src) / 2 * 2

	if limit > 0 {
		SigmoidNEON(dst[:limit], src[:limit])
	}

	scalarSigmoid(dst[limit:], src[limit:])
}
