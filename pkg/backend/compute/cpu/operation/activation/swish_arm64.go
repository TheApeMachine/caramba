//go:build arm64

package activation

import "fmt"

//go:noescape
func SwishNEON(dst, src []float64)

func swishKernel(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf("swishKernel: dst shorter than src (dst=%d src=%d)", len(dst), len(src)))
	}

	SwishNEON(dst[:len(src)], src)
}
