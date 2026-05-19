package kernels

import (
	"math"
	"math/rand"
	"testing"
)

func randFloat32Slice(elementCount int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	slice := make([]float32, elementCount)

	for index := range slice {
		slice[index] = float32(rng.NormFloat64()) * 0.1
	}

	return slice
}

func assertFloat32SlicesNear(
	testing *testing.T,
	got, want []float32,
	tolerance float64,
) {
	testing.Helper()

	if len(got) != len(want) {
		testing.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}

	for index := range got {
		diff := math.Abs(float64(got[index] - want[index]))

		if diff > tolerance {
			testing.Fatalf(
				"lane %d got=%g want=%g diff=%g",
				index, got[index], want[index], diff,
			)
		}
	}
}
