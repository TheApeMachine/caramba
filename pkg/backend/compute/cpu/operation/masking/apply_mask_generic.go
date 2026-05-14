//go:build !amd64 && !arm64

package masking

func applyMaskKernel(dst, scores, mask []float64) {
	applyMaskScalar(dst, scores, mask)
}
