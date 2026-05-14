package shape

func splitGenericKernel(dst, src []float64, outer, dimSize, splitSize, inner int) {
	numChunks := dimSize / splitSize
	chunkElems := outer * splitSize * inner

	for chunk := 0; chunk < numChunks; chunk++ {
		dstOffset := chunk * chunkElems

		for outerIndex := 0; outerIndex < outer; outerIndex++ {
			srcOffset := (outerIndex*dimSize + chunk*splitSize) * inner
			elementCount := splitSize * inner
			copyBlock(dst[dstOffset:dstOffset+elementCount], src[srcOffset:srcOffset+elementCount])
			dstOffset += elementCount
		}
	}
}
