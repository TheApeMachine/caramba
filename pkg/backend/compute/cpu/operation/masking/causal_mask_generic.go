//go:build !amd64 && !arm64

package masking

func causalMaskKernel(dst []float64, seqLen int) {
	causalMaskScalar(dst, seqLen)
}
