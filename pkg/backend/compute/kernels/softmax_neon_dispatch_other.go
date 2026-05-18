//go:build !arm64

package kernels

import "math"

func softmaxRowFillExpsNative(row []float32, outRow []float32, maxValue float32) float32 {
	var sum float32
	for index, candidate := range row {
		shifted := float32(math.Exp(float64(candidate - maxValue)))
		outRow[index] = shifted
		sum += shifted
	}
	return sum
}
