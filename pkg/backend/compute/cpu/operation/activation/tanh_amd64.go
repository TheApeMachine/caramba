//go:build amd64

package activation

//go:noescape
func TanhAVX2(dst, src []float64)

//go:noescape
func TanhSSE2(dst, src []float64)

func applyTanh(dst, src []float64) {
	elementCount := len(src)

	if len(dst) < elementCount {
		elementCount = len(dst)
	}

	width := 2
	vectorTanh := TanhSSE2

	if useAVX2 {
		width = 4
		vectorTanh = TanhAVX2
	}

	limit := elementCount / width * width

	if limit > 0 {
		vectorTanh(dst[:limit], src[:limit])
	}

	scalarTanh(dst[limit:elementCount], src[limit:elementCount])
}
