//go:build !amd64 && !arm64

package cpu

import "math"

func AbsFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = float32(math.Abs(float64(value)))
	}
}

func NegFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = -value
	}
}

func SqrtFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = float32(math.Sqrt(float64(value)))
	}
}

func ReluFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = 0

		if value > 0 {
			dst[index] = value
		}
	}
}
