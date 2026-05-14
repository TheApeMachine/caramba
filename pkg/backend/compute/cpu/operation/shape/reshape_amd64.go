//go:build amd64

package shape

func reshapeKernel(dst, src []float64) {
	if useAVX2 {
		CopyAVX2(dst, src)

		return
	}

	CopySSE2(dst, src)
}
