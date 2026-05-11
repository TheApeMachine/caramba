//go:build arm64

package activation

//go:noescape
func LeakyReLUNEON(dst, src []float64, alpha float64)

func applyLeakyReLU(dst, src []float64, alpha float64) {
	limit := len(src) / 2 * 2

	if limit > 0 {
		LeakyReLUNEON(dst[:limit], src[:limit], alpha)
	}

	scalarLeakyReLU(dst[limit:], src[limit:], alpha)
}
