//go:build amd64

package shape

//go:noescape
func viewAsHeadsCopyAVX2(dst, src []float64)

//go:noescape
func viewAsHeadsCopySSE2(dst, src []float64)

func viewAsHeadsKernel(dst, src []float64, batch, tokens, numHeads, headDim int) {
	dimension := numHeads * headDim

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for headIndex := 0; headIndex < numHeads; headIndex++ {
			for tokenIndex := 0; tokenIndex < tokens; tokenIndex++ {
				srcOffset := (batchIndex*tokens+tokenIndex)*dimension + headIndex*headDim
				dstOffset := ((batchIndex*numHeads+headIndex)*tokens + tokenIndex) * headDim
				srcRow := src[srcOffset : srcOffset+headDim]
				dstRow := dst[dstOffset : dstOffset+headDim]

				if useAVX2 {
					viewAsHeadsCopyAVX2(dstRow, srcRow)
				} else {
					viewAsHeadsCopySSE2(dstRow, srcRow)
				}
			}
		}
	}
}
