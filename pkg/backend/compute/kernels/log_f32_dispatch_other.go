//go:build !arm64

package kernels

import "math"

func logFloat32Native(dst, src []float32) {
	for i, x := range src {
		dst[i] = float32(math.Log(float64(x)))
	}
}
