//go:build arm64

package activation

import "fmt"

//go:noescape
func SwiGLUNEON(dst, src []float64)

func applySwiGLU(dst, src []float64) {
	if len(src) != 2*len(dst) {
		panic(fmt.Sprintf(
			"applySwiGLU: expected len(src)==2*len(dst), got len(dst)=%d len(src)=%d",
			len(dst), len(src),
		))
	}

	SwiGLUNEON(dst, src)
}
