//go:build arm64

package activation

//go:noescape
func SwiGLUNEON(dst, src []float64)

func applySwiGLU(dst, src []float64) {
	SwiGLUNEON(dst, src)
}
