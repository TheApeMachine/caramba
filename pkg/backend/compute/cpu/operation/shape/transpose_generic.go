//go:build !amd64 && !arm64

package shape

func copyBlock(dst, src []float64) {
	copy(dst, src)
}

func transposeKernel(dst, src []float64, shape []int, dim0, dim1 int) {
	transposeGenericKernel(dst, src, shape, dim0, dim1)
}
