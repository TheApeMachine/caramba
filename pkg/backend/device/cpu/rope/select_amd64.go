//go:build amd64

package rope

import "golang.org/x/sys/cpu"

func RopePairsNative(out, in, cosBuf, sinBuf []float32) {
	halfDim := len(cosBuf)

	if halfDim == 0 {
		return
	}

	if cpu.X86.HasAVX512F {
		blockPairs := halfDim & ^7

		if blockPairs > 0 {
			RopePairsFloat32AVX512Asm(
				&out[0], &in[0], &cosBuf[0], &sinBuf[0], blockPairs,
			)
		}

		for pairIndex := blockPairs; pairIndex < halfDim; pairIndex++ {
			cos := cosBuf[pairIndex]
			sin := sinBuf[pairIndex]
			even := in[2*pairIndex]
			odd := in[2*pairIndex+1]
			out[2*pairIndex] = even*cos - odd*sin
			out[2*pairIndex+1] = even*sin + odd*cos
		}

		return
	}

	RopePairsGeneric(out, in, cosBuf, sinBuf)
}
