//go:build !arm64

package cpu

import "math"

func LogFloat32Native(dst, src []float32) {
	for i, x := range src {
		dst[i] = float32(math.Log(float64(x)))
	}
}
