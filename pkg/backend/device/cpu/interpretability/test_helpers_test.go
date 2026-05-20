package interpretability

import (
	"math"
	"math/rand"
	"testing"
)

func randomSteerVectors(length int, seed int64) (base, direction []float32) {
	rng := rand.New(rand.NewSource(seed))
	base = make([]float32, length)
	direction = make([]float32, length)

	for index := range base {
		base[index] = float32((rng.Float64() - 0.5) * 4)
		direction[index] = float32((rng.Float64() - 0.5) * 4)
	}

	return base, direction
}

func assertFloat32SliceEqual(testingObject *testing.T, got, want []float32) {
	testingObject.Helper()

	if len(got) != len(want) {
		testingObject.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}

	for index := range got {
		if got[index] != want[index] {
			if math.Abs(float64(got[index]-want[index])) > 1e-6 {
				testingObject.Fatalf(
					"index %d: got %g want %g",
					index, got[index], want[index],
				)
			}
		}
	}
}
