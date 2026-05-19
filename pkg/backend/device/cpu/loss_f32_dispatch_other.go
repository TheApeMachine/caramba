//go:build !arm64

package cpu

import "math"

func MseSumFloat32Native(predictions, targets []float32) float32 {
	var sum float64

	for index, value := range predictions {
		delta := float64(value - targets[index])
		sum += delta * delta
	}

	return float32(sum)
}

func MaeSumFloat32Native(predictions, targets []float32) float32 {
	var sum float64

	for index, value := range predictions {
		sum += math.Abs(float64(value - targets[index]))
	}

	return float32(sum)
}

func L1NormFloat32Native(values []float32) float32 {
	var sum float64

	for _, value := range values {
		sum += math.Abs(float64(value))
	}

	return float32(sum)
}
