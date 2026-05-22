package diffusion

import (
	"fmt"
	"math"
)

/*
VectorStats summarizes one float32 vector.
*/
type VectorStats struct {
	Length float64
	Mean   float64
	Std    float64
	Min    float32
	Max    float32
}

/*
CompareVectors reports L2 and max-abs difference between two equal-length vectors.
*/
func CompareVectors(left []float32, right []float32) (l2 float64, maxAbs float32, err error) {
	if len(left) != len(right) {
		return 0, 0, fmt.Errorf("diffusion: vector lengths %d vs %d", len(left), len(right))
	}

	if len(left) == 0 {
		return 0, 0, nil
	}

	var sumSquares float64

	for index := range left {
		delta := float64(left[index] - right[index])
		sumSquares += delta * delta

		absDelta := float32(math.Abs(delta))

		if absDelta > maxAbs {
			maxAbs = absDelta
		}
	}

	return math.Sqrt(sumSquares), maxAbs, nil
}

/*
StatsVector computes norm and distribution stats for one vector.
*/
func StatsVector(values []float32) VectorStats {
	if len(values) == 0 {
		return VectorStats{}
	}

	var sum float64
	var sumSquares float64
	minValue := values[0]
	maxValue := values[0]

	for _, value := range values {
		sum += float64(value)
		sumSquares += float64(value) * float64(value)

		if value < minValue {
			minValue = value
		}

		if value > maxValue {
			maxValue = value
		}
	}

	mean := sum / float64(len(values))
	variance := sumSquares/float64(len(values)) - mean*mean

	if variance < 0 {
		variance = 0
	}

	return VectorStats{
		Length: math.Sqrt(sumSquares),
		Mean:   mean,
		Std:    math.Sqrt(variance),
		Min:    minValue,
		Max:    maxValue,
	}
}

/*
ZeroVector returns a zero-filled vector with the requested length.
*/
func ZeroVector(length int) []float32 {
	return make([]float32, length)
}
