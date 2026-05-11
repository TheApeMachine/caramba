//go:build amd64

package activation

import "fmt"

//go:noescape
func LeakyReLUAVX2(dst, src []float64, alpha float64)

//go:noescape
func LeakyReLUSSE2(dst, src []float64, alpha float64)

// applyLeakyReLU writes len(src) elements into dst; dst and src must have equal length.
func applyLeakyReLU(dst, src []float64, alpha float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("applyLeakyReLU: dst and src length mismatch: dst=%d src=%d", len(dst), len(src)))
	}

	width := 2
	vectorLeakyReLU := LeakyReLUSSE2

	if useAVX2 {
		width = 4
		vectorLeakyReLU = LeakyReLUAVX2
	}

	limit := len(src) / width * width

	if limit > 0 {
		vectorLeakyReLU(dst[:limit], src[:limit], alpha)
	}

	scalarLeakyReLU(dst[limit:], src[limit:], alpha)
}
