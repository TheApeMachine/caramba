//go:build amd64

package activation

//go:noescape
func LeakyReLUAVX2(dst, src []float64, alpha float64)

//go:noescape
func LeakyReLUSSE2(dst, src []float64, alpha float64)

func applyLeakyReLU(dst, src []float64, alpha float64) {
	width := 2

	if useAVX2 {
		width = 4
		limit := len(src) / width * width

		if limit > 0 {
			LeakyReLUAVX2(dst[:limit], src[:limit], alpha)
		}

		scalarLeakyReLU(dst[limit:], src[limit:], alpha)
		return
	}

	limit := len(src) / width * width

	if limit > 0 {
		LeakyReLUSSE2(dst[:limit], src[:limit], alpha)
	}

	scalarLeakyReLU(dst[limit:], src[limit:], alpha)
}
