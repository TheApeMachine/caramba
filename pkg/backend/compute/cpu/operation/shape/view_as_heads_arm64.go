//go:build arm64

package shape

//go:noescape
func viewAsHeadsCopyNEON(dst, src []float64)

func viewAsHeadsKernel(dst, src []float64, batch, tokens, numHeads, headDim int) {
	dimension := numHeads * headDim

	for batchIndex := range batch {
		for headIndex := range numHeads {
			for tokenIndex := range tokens {
				srcOffset := (batchIndex*tokens+tokenIndex)*dimension + headIndex*headDim
				dstOffset := ((batchIndex*numHeads+headIndex)*tokens + tokenIndex) * headDim
				viewAsHeadsCopyNEON(
					dst[dstOffset:dstOffset+headDim],
					src[srcOffset:srcOffset+headDim],
				)
			}
		}
	}
}
