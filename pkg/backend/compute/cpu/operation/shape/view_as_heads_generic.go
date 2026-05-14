//go:build !amd64 && !arm64

package shape

func viewAsHeadsKernel(dst, src []float64, batch, tokens, numHeads, headDim int) {
	dimension := numHeads * headDim

	for batchIndex := range batch {
		for headIndex := range numHeads {
			for tokenIndex := range tokens {
				srcOffset := (batchIndex*tokens+tokenIndex)*dimension + headIndex*headDim
				dstOffset := ((batchIndex*numHeads+headIndex)*tokens + tokenIndex) * headDim

				copy(dst[dstOffset:dstOffset+headDim], src[srcOffset:srcOffset+headDim])
			}
		}
	}
}
