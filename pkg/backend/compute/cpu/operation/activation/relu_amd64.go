//go:build amd64

package activation

//go:noescape
func ReLUAVX2(dst, src []float64)

//go:noescape
func ReLUSSE2(dst, src []float64)

func applyReLU(dst, src []float64) {
	width := 2

	if useAVX2 {
		width = 4
		limit := len(src) / width * width

		if limit > 0 {
			ReLUAVX2(dst[:limit], src[:limit])
		}

		scalarReLU(dst[limit:], src[limit:])
		return
	}

	limit := len(src) / width * width

	if limit > 0 {
		ReLUSSE2(dst[:limit], src[:limit])
	}

	scalarReLU(dst[limit:], src[limit:])
}
