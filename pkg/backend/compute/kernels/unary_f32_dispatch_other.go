//go:build !amd64 && !arm64

package kernels

import "math"

func absFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = float32(math.Abs(float64(value)))
	}
}

func negFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = -value
	}
}

func sqrtFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = float32(math.Sqrt(float64(value)))
	}
}

func reluFloat32Native(dst, src []float32) {
	for index, value := range src {
		dst[index] = 0

		if value > 0 {
			dst[index] = value
		}
	}
}
