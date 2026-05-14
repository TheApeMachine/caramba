//go:build amd64

package activation

import "fmt"

//go:noescape
func SwishAVX2(dst, src []float64)

//go:noescape
func SwishSSE2(dst, src []float64)

//go:noescape
func swishScalarAMD64(dst, src []float64)

func swishKernel(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf("swishKernel: dst shorter than src (dst=%d src=%d)", len(dst), len(src)))
	}

	elementCount := len(src)
	offset := 0

	if useAVX2 {
		limit := elementCount / 4 * 4

		if limit > 0 {
			SwishAVX2(dst[:limit], src[:limit])
			offset = limit
		}
	}

	limit := offset + (elementCount-offset)/2*2

	if limit > offset {
		SwishSSE2(dst[offset:limit], src[offset:limit])
		offset = limit
	}

	if offset < elementCount {
		swishScalarAMD64(dst[offset:elementCount], src[offset:elementCount])
	}
}
