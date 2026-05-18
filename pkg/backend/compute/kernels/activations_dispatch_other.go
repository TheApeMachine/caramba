//go:build !arm64

package kernels

import "math"

func sigmoidFloat32Native(dst, src []float32) {
	for i, x := range src {
		dst[i] = 1 / (1 + float32(math.Exp(float64(-x))))
	}
}

func siluFloat32Native(dst, src []float32) {
	for i, x := range src {
		dst[i] = x / (1 + float32(math.Exp(float64(-x))))
	}
}

func tanhFloat32Native(dst, src []float32) {
	for i, x := range src {
		dst[i] = float32(math.Tanh(float64(x)))
	}
}
