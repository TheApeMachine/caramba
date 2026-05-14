//go:build amd64

package shape

//go:noescape
func mergeHeadsCopyAVX2(dst, src []float64)

//go:noescape
func mergeHeadsCopySSE2(dst, src []float64)

func mergeHeadsKernel(dst, src []float64, batch, numHeads, tokens, headDim int) {
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for tokenIndex := 0; tokenIndex < tokens; tokenIndex++ {
			for headIndex := 0; headIndex < numHeads; headIndex++ {
				srcOffset := ((batchIndex*numHeads+headIndex)*tokens + tokenIndex) * headDim
				dstOffset := ((batchIndex*tokens+tokenIndex)*numHeads + headIndex) * headDim
				srcRow := src[srcOffset : srcOffset+headDim]
				dstRow := dst[dstOffset : dstOffset+headDim]

				if useAVX2 {
					mergeHeadsCopyAVX2(dstRow, srcRow)
				} else {
					mergeHeadsCopySSE2(dstRow, srcRow)
				}
			}
		}
	}
}
