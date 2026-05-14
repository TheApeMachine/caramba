//go:build arm64

package shape

//go:noescape
func mergeHeadsCopyNEON(dst, src []float64)

func mergeHeadsKernel(dst, src []float64, batch, numHeads, tokens, headDim int) {
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for tokenIndex := 0; tokenIndex < tokens; tokenIndex++ {
			for headIndex := 0; headIndex < numHeads; headIndex++ {
				srcOffset := ((batchIndex*numHeads+headIndex)*tokens + tokenIndex) * headDim
				dstOffset := ((batchIndex*tokens+tokenIndex)*numHeads + headIndex) * headDim
				mergeHeadsCopyNEON(
					dst[dstOffset:dstOffset+headDim],
					src[srcOffset:srcOffset+headDim],
				)
			}
		}
	}
}
