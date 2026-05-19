//go:build arm64

package rope

func RopePairsNative(out, in, cosBuf, sinBuf []float32) {
	halfDim := len(cosBuf)
	if halfDim == 0 {
		return
	}

	// NEON handles 4 pairs per iteration.
	blockPairs := halfDim & ^3
	if blockPairs > 0 {
		RopePairsNEONAsm(&out[0], &in[0], &cosBuf[0], &sinBuf[0], blockPairs)
	}

	// Scalar tail for the remaining 0..3 pairs.
	for pairIndex := blockPairs; pairIndex < halfDim; pairIndex++ {
		cos := cosBuf[pairIndex]
		sin := sinBuf[pairIndex]
		even := in[2*pairIndex]
		odd := in[2*pairIndex+1]
		out[2*pairIndex] = even*cos - odd*sin
		out[2*pairIndex+1] = even*sin + odd*cos
	}
}
