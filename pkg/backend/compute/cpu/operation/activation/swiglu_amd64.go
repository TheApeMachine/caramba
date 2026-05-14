//go:build amd64

package activation

import "fmt"

//go:noescape
func SwiGLUAVX2(dst, src []float64)

//go:noescape
func SwiGLUSSE2(dst, src []float64)

func swigluKernel(dst, src []float64) {
	if len(src) != 2*len(dst) {
		panic(fmt.Sprintf(
			"swigluKernel: expected len(src)==2*len(dst), got len(dst)=%d len(src)=%d",
			len(dst), len(src),
		))
	}

	if len(dst)%2 != 0 {
		scalarSwiGLU(dst, src)

		return
	}

	if useAVX2 && len(dst)%4 == 0 {
		SwiGLUAVX2(dst, src)

		return
	}

	SwiGLUSSE2(dst, src)
}
