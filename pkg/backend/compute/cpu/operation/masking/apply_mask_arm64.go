//go:build arm64

package masking

//go:noescape
func ApplyMaskNEON(dst, scores, mask []float64)

func applyMaskKernel(dst, scores, mask []float64) {
	ApplyMaskNEON(dst, scores, mask)
}
