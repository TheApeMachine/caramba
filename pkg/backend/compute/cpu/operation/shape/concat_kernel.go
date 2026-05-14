package shape

func concatGenericKernel(dst []float64, inputs [][]float64, outer, dimSize, inner int) {
	dstOffset := 0
	blockSize := dimSize * inner

	for outerIndex := 0; outerIndex < outer; outerIndex++ {
		for _, src := range inputs {
			srcOffset := outerIndex * blockSize
			block := src[srcOffset : srcOffset+blockSize]
			copyBlock(dst[dstOffset:dstOffset+len(block)], block)
			dstOffset += len(block)
		}
	}
}
