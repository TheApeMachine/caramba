//go:build arm64

package masking

//go:noescape
func ApplyMaskNEON(dst, scores, mask []float64)

func applyMaskAdd(dst, scores, mask []float64) {
	ApplyMaskNEON(dst, scores, mask)
}
