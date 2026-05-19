//go:build !arm64

package cpu

import "math"

func ExpFloat32Native(dst, src []float32) {
	for i, x := range src {
		dst[i] = float32(math.Exp(float64(x)))
	}
}
