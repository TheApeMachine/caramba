//go:build arm64

package activation

//go:noescape
func TanhNEON(dst, src []float64)

func applyTanh(dst, src []float64) {
	limit := len(src) / 2 * 2

	if limit > 0 {
		TanhNEON(dst[:limit], src[:limit])
	}

	scalarTanh(dst[limit:], src[limit:])
}
