//go:build !amd64 && !arm64

package cpu

/*
Scalar fallback for architectures without a SIMD path yet. The
implementation matches the Go reference so behavior is identical to
the native paths up to floating-point rounding.
*/

func AddFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] + right[index]
	}
}

func SubFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] - right[index]
	}
}

func MulFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] * right[index]
	}
}

func DivFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = left[index] / right[index]
	}
}

func MaxFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = right[index]

		if left[index] > right[index] {
			dst[index] = left[index]
		}
	}
}

func MinFloat32Native(dst, left, right []float32) {
	for index := range dst {
		dst[index] = right[index]

		if left[index] < right[index] {
			dst[index] = left[index]
		}
	}
}
