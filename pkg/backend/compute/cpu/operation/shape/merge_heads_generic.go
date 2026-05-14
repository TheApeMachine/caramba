//go:build !amd64 && !arm64

package shape

func mergeHeadsKernel(dst, src []float64, batch, numHeads, tokens, headDim int) {
	for batchIndex := range batch {
		for tokenIndex := range tokens {
			for headIndex := range numHeads {
				srcOffset := ((batchIndex*numHeads+headIndex)*tokens + tokenIndex) * headDim
				dstOffset := ((batchIndex*tokens+tokenIndex)*numHeads + headIndex) * headDim

				copy(dst[dstOffset:dstOffset+headDim], src[srcOffset:srcOffset+headDim])
			}
		}
	}
}
