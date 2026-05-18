//go:build !arm64

package kernels

import "math"

func mseSumFloat32Native(predictions, targets []float32) float32 {
	var sum float64

	for index, value := range predictions {
		delta := float64(value - targets[index])
		sum += delta * delta
	}

	return float32(sum)
}

func maeSumFloat32Native(predictions, targets []float32) float32 {
	var sum float64

	for index, value := range predictions {
		sum += math.Abs(float64(value - targets[index]))
	}

	return float32(sum)
}

func l1NormFloat32Native(values []float32) float32 {
	var sum float64

	for _, value := range values {
		sum += math.Abs(float64(value))
	}

	return float32(sum)
}
