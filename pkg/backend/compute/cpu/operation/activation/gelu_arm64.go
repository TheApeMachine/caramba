//go:build arm64

package activation

//go:noescape
func GeLUNEON(dst, x []float64)

func geluKernel(dst, src []float64) {
	limit := len(src) / 2 * 2

	if limit > 0 {
		GeLUNEON(dst[:limit], src[:limit])
	}

	scalarGeLU(dst[limit:], src[limit:])
}
