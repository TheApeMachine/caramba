//go:build amd64

package activation

//go:noescape
func TanhAVX2(dst, src []float64)

//go:noescape
func TanhSSE2(dst, src []float64)

func applyTanh(dst, src []float64) {
	width := 2

	if useAVX2 {
		width = 4
		limit := len(src) / width * width

		if limit > 0 {
			TanhAVX2(dst[:limit], src[:limit])
		}

		scalarTanh(dst[limit:], src[limit:])
		return
	}

	limit := len(src) / width * width

	if limit > 0 {
		TanhSSE2(dst[:limit], src[:limit])
	}

	scalarTanh(dst[limit:], src[limit:])
}
