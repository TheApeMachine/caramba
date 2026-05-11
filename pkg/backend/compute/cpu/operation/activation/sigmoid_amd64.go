//go:build amd64

package activation

//go:noescape
func SigmoidAVX2(dst, src []float64)

//go:noescape
func SigmoidSSE2(dst, src []float64)

func applySigmoid(dst, src []float64) {
	width := 2

	if useAVX2 {
		width = 4
		limit := len(src) / width * width

		if limit > 0 {
			SigmoidAVX2(dst[:limit], src[:limit])
		}

		scalarSigmoid(dst[limit:], src[limit:])
		return
	}

	limit := len(src) / width * width

	if limit > 0 {
		SigmoidSSE2(dst[:limit], src[:limit])
	}

	scalarSigmoid(dst[limit:], src[limit:])
}
