//go:build arm64

package activation

import "fmt"

//go:noescape
func ReLUNEON(dst, src []float64)

func applyReLU(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf("applyReLU: dst shorter than src (dst=%d src=%d)", len(dst), len(src)))
	}

	limit := len(src) / 2 * 2

	if limit > 0 {
		ReLUNEON(dst[:limit], src[:limit])
	}

	scalarReLU(dst[limit:], src[limit:])
}
