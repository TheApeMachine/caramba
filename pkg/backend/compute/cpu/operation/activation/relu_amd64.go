//go:build amd64

package activation

import "fmt"

//go:noescape
func ReLUAVX2(dst, src []float64)

//go:noescape
func ReLUSSE2(dst, src []float64)

func applyReLU(dst, src []float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("applyReLU: dst and src length mismatch: dst=%d src=%d", len(dst), len(src)))
	}

	width := 2
	vectorReLU := ReLUSSE2

	if useAVX2 {
		width = 4
		vectorReLU = ReLUAVX2
	}

	limit := len(src) / width * width

	if limit > 0 {
		vectorReLU(dst[:limit], src[:limit])
	}

	scalarReLU(dst[limit:], src[limit:])
}
