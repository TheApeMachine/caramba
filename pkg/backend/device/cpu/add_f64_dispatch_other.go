//go:build !arm64

package cpu

func AddFloat64Native(dst, left, right []float64) {
	for index := range dst {
		dst[index] = left[index] + right[index]
	}
}
