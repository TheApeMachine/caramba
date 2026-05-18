//go:build !arm64

package kernels

import "math"

func geluTanhFloat32Native(dst, src []float32) {
	for i, x := range src {
		v := float64(x)
		inner := 0.7978845608028654 * (v + 0.044715*v*v*v)
		dst[i] = float32(0.5 * v * (1 + math.Tanh(inner)))
	}
}
