//go:build !arm64

package kernels

func addFloat64Native(dst, left, right []float64) {
	for index := range dst {
		dst[index] = left[index] + right[index]
	}
}
