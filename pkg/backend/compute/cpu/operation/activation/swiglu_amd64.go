//go:build amd64

package activation

//go:noescape
func SwiGLUAVX2(dst, src []float64)

//go:noescape
func SwiGLUSSE2(dst, src []float64)

func applySwiGLU(dst, src []float64) {
	if useAVX2 {
		SwiGLUAVX2(dst, src)
	} else {
		SwiGLUSSE2(dst, src)
	}
}
