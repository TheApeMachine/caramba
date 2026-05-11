//go:build arm64

package activation

import "fmt"

//go:noescape
func SigmoidNEON(dst, src []float64)

func applySigmoid(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf("applySigmoid: dst shorter than src (dst=%d src=%d)", len(dst), len(src)))
	}

	limit := len(src) / 2 * 2

	if limit > 0 {
		SigmoidNEON(dst[:limit], src[:limit])
	}

	scalarSigmoid(dst[limit:], src[limit:])
}
