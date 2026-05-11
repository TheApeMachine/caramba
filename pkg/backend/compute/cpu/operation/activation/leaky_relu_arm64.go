//go:build arm64

package activation

import "fmt"

//go:noescape
func LeakyReLUNEON(dst, src []float64, alpha float64)

// applyLeakyReLU writes len(src) elements into dst; dst and src must have equal length.
func applyLeakyReLU(dst, src []float64, alpha float64) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("applyLeakyReLU: dst and src length mismatch: dst=%d src=%d", len(dst), len(src)))
	}

	// NEON processes float64 two lanes at a time in 128-bit registers (pairwise);
	// LeakyReLUNEON therefore requires an even-length prefix. An odd trailing
	// element uses scalarLeakyReLU.
	limit := len(src) / 2 * 2

	if limit > 0 {
		LeakyReLUNEON(dst[:limit], src[:limit], alpha)
	}

	scalarLeakyReLU(dst[limit:], src[limit:], alpha)
}
