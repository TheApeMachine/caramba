//go:build arm64

package activation

import "fmt"

//go:noescape
func swishNEON(dst, src []float64)

/*
SwishKernel applies the Swish activation to src and writes the result to dst.
dst must be at least as long as src; otherwise SwishKernel panics. The operation
is safe for in-place use when dst and src share storage, and delegates the core
computation to swishNEON.
*/
func SwishKernel(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf("SwishKernel: dst shorter than src (dst=%d src=%d)", len(dst), len(src)))
	}

	swishNEON(dst[:len(src)], src)
}
