//go:build !amd64 && !arm64

package quant

import "math"

func QuantInt8Native(dst []int8, src []float32, scale float32, zeroPoint int8) {
	for index, value := range src {
		scaled := math.Round(float64(value/scale)) + float64(zeroPoint)

		switch {
		case scaled > float64(math.MaxInt8):
			dst[index] = math.MaxInt8
		case scaled < float64(math.MinInt8):
			dst[index] = math.MinInt8
		default:
			dst[index] = int8(scaled)
		}
	}
}
