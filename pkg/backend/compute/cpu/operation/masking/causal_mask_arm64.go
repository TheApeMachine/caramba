//go:build arm64

package masking

//go:noescape
func CausalMaskNEON(dst []float64, seqLen int)

func causalMaskKernel(dst []float64, seqLen int) {
	CausalMaskNEON(dst, seqLen)
}
