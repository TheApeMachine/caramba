//go:build arm64

package masking

import "math"

//go:noescape
func CausalMaskNEON(dst []float64, seqLen int)

func applyCausalMask(dst []float64, seqLen int) {
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j <= i {
				dst[i*seqLen+j] = 0.0
			} else {
				dst[i*seqLen+j] = math.Inf(-1)
			}
		}
	}
}
