//go:build arm64

package activation

import "fmt"

//go:noescape
func TanhNEON(dst, src []float64)

func applyTanh(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf("applyTanh: dst shorter than src (dst=%d src=%d)", len(dst), len(src)))
	}

	limit := len(src) / 2 * 2

	if limit > 0 {
		TanhNEON(dst[:limit], src[:limit])
	}

	scalarTanh(dst[limit:], src[limit:])
}
