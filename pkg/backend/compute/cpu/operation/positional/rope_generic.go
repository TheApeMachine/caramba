//go:build !amd64 && !arm64

package positional

func ropeAdvanceRow(cosCur, sinCur, cosStep, sinStep []float64) {
	for index := range cosCur {
		cosValue := cosCur[index]
		sinValue := sinCur[index]

		cosCur[index] = cosValue*cosStep[index] - sinValue*sinStep[index]
		sinCur[index] = sinValue*cosStep[index] + cosValue*sinStep[index]
	}
}

func ropeKernel(
	dst, src, cosTable, sinTable []float64,
	batch, numHeads, seqLen, numPairs int,
) {
	headDim := numPairs * 2

	for batchIndex := range batch {
		for headIndex := range numHeads {
			for tokenIndex := range seqLen {
				offset := ((batchIndex*numHeads+headIndex)*seqLen + tokenIndex) * headDim
				cosOffset := tokenIndex * numPairs

				for pairIndex := range numPairs {
					inputOffset := offset + pairIndex*2
					tableOffset := cosOffset + pairIndex
					x0 := src[inputOffset]
					x1 := src[inputOffset+1]
					cosValue := cosTable[tableOffset]
					sinValue := sinTable[tableOffset]

					dst[inputOffset] = x0*cosValue - x1*sinValue
					dst[inputOffset+1] = x0*sinValue + x1*cosValue
				}
			}
		}
	}
}

func ropeKernelHalf(
	dst, src, cosTable, sinTable []float64,
	batch, numHeads, seqLen, numPairs int,
) {
	headDim := numPairs * 2

	for batchIndex := range batch {
		for headIndex := range numHeads {
			for tokenIndex := range seqLen {
				offset := ((batchIndex*numHeads+headIndex)*seqLen + tokenIndex) * headDim
				cosOffset := tokenIndex * numPairs

				for pairIndex := range numPairs {
					firstOffset := offset + pairIndex
					secondOffset := firstOffset + numPairs
					tableOffset := cosOffset + pairIndex
					x0 := src[firstOffset]
					x1 := src[secondOffset]
					cosValue := cosTable[tableOffset]
					sinValue := sinTable[tableOffset]

					dst[firstOffset] = x0*cosValue - x1*sinValue
					dst[secondOffset] = x0*sinValue + x1*cosValue
				}
			}
		}
	}
}
