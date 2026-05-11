//go:build arm64

package activation

//go:noescape
func ReLUNEON(dst, src []float64)

func applyReLU(dst, src []float64) {
	limit := len(src) / 2 * 2

	if limit > 0 {
		ReLUNEON(dst[:limit], src[:limit])
	}

	scalarReLU(dst[limit:], src[limit:])
}
