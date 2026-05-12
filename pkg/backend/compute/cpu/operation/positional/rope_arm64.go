//go:build arm64

package positional

//go:noescape
func RoPENEON(dst, src, cosTable, sinTable []float64, numPairs int)

//go:noescape
func ropeAdvanceRowNEON(cosCur, sinCur, cosStep, sinStep []float64)

func ropeAdvanceRow(cosCur, sinCur, cosStep, sinStep []float64) {
	ropeAdvanceRowNEON(cosCur, sinCur, cosStep, sinStep)
}

func applyRoPE(dst, src, cosTable, sinTable []float64, batch, numHeads, seqLen, numPairs int) {
	headDim := numPairs * 2
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for t := 0; t < seqLen; t++ {
				offset := ((b*numHeads+h)*seqLen+t) * headDim
				xSlice := src[offset : offset+headDim]
				dSlice := dst[offset : offset+headDim]
				cosSlice := cosTable[t*numPairs : (t+1)*numPairs]
				sinSlice := sinTable[t*numPairs : (t+1)*numPairs]
				RoPENEON(dSlice, xSlice, cosSlice, sinSlice, numPairs)
			}
		}
	}
}
