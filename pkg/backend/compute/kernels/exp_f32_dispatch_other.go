//go:build !arm64

package kernels

import "math"

func expFloat32Native(dst, src []float32) {
	for i, x := range src {
		dst[i] = float32(math.Exp(float64(x)))
	}
}
