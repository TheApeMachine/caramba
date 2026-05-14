//go:build arm64

package activation

import "fmt"

//go:noescape
func swishNEON(dst, src []float64)

func SwishKernel(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf("SwishKernel: dst shorter than src (dst=%d src=%d)", len(dst), len(src)))
	}

	swishNEON(dst[:len(src)], src)
}
