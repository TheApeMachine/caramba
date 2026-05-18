//go:build !amd64 && !arm64

package kernels

/*
Scalar fallback for architectures without a SIMD path yet. The
implementation matches the Go reference so behavior is identical to
the native paths up to floating-point rounding.
*/

func addFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] + right[index]
	}
}

func subFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] - right[index]
	}
}

func mulFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] * right[index]
	}
}

func divFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] / right[index]
	}
}

func maxFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = right[index]

		if left[index] > right[index] {
			dst[index] = left[index]
		}
	}
}

func minFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = right[index]

		if left[index] < right[index] {
			dst[index] = left[index]
		}
	}
}
