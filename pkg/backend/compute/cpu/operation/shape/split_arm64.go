//go:build arm64

package shape

func splitKernel(dst, src []float64, outer, dimSize, splitSize, inner int) {
	splitGenericKernel(dst, src, outer, dimSize, splitSize, inner)
}
