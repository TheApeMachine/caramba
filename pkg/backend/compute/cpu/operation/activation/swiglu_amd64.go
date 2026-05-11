//go:build amd64

package activation

//go:noescape
func SwiGLUAVX2(dst, src []float64)

//go:noescape
func SwiGLUSSE2(dst, src []float64)

func applySwiGLU(dst, src []float64) {
	if useAVX2 && len(dst)%4 == 0 {
		SwiGLUAVX2(dst, src)
		return
	}

	if !useAVX2 && len(dst)%2 == 0 {
		SwiGLUSSE2(dst, src)
		return
	}

	scalarSwiGLU(dst, src)
}
