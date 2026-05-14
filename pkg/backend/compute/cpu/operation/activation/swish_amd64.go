//go:build amd64

package activation

import "fmt"

/*
SwishAVX2 applies the Swish activation to src and writes the results to dst.
dst and src are float64 slices and dst must have the same length as src.
The caller must only invoke this stub when AVX2 is supported and the slice
length is a multiple of four. The routine writes dst in place and allocates
nothing.
*/
//go:noescape
func SwishAVX2(dst, src []float64)

/*
SwishSSE2 applies the Swish activation to src and writes the results to dst.
dst and src are float64 slices and dst must have the same length as src.
The caller must only invoke this stub when SSE2 is supported and the slice
length is a multiple of two. The routine writes dst in place and allocates
nothing.
*/
//go:noescape
func SwishSSE2(dst, src []float64)

//go:noescape
func swishScalarAMD64(dst, src []float64)

/*
SwishKernel applies the Swish activation to src and writes the result to dst.
dst and src are float64 slices, and dst must be at least as long as src. The
operation is safe for in-place use when dst and src reference the same backing
array. Platform-optimized AVX2 or SSE2 implementations are selected
automatically where available, with scalar code handling the remaining tail.
*/
func SwishKernel(dst, src []float64) {
	if len(dst) < len(src) {
		panic(fmt.Sprintf("SwishKernel: dst shorter than src (dst=%d src=%d)", len(dst), len(src)))
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
