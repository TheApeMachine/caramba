//go:build !arm64

package rope

func RopePairsNative(out, in, cosBuf, sinBuf []float32) {
	for pairIndex, cos := range cosBuf {
		sin := sinBuf[pairIndex]
		even := in[2*pairIndex]
		odd := in[2*pairIndex+1]
		out[2*pairIndex] = even*cos - odd*sin
		out[2*pairIndex+1] = even*sin + odd*cos
	}
}
