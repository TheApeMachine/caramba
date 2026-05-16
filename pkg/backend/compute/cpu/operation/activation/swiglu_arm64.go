//go:build arm64

package activation

import "fmt"

//go:noescape
func SwiGLUNEON(dst, gates, values []float64)

func swigluKernel(dst, src []float64) {
	if len(src) != 2*len(dst) {
		panic(fmt.Sprintf(
			"swigluKernel: expected len(src)==2*len(dst), got len(dst)=%d len(src)=%d",
			len(dst), len(src),
		))
	}

	half := len(dst)
	limit := half / 2 * 2

	if limit > 0 {
		SwiGLUNEON(dst[:limit], src[:limit], src[half:half+limit])
	}

	if limit < half {
		gate := src[limit]
		value := src[half+limit]
		dst[limit] = gate * scalarSigmoidAt(gate) * value
	}
}
