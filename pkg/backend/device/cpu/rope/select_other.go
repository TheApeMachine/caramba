//go:build !amd64 && !arm64

package rope

func RopePairsNative(out, in, cosBuf, sinBuf []float32) {
	RopePairsGeneric(out, in, cosBuf, sinBuf)
}
