package model_editing

import (
	"math"
	"math/rand"
	"testing"
)

func randomGraftVectors(length int, seed int64) (weights, injection []float32) {
	rng := rand.New(rand.NewSource(seed))
	weights = make([]float32, length)
	injection = make([]float32, length)

	for index := range weights {
		weights[index] = float32((rng.Float64() - 0.5) * 8)
		injection[index] = float32((rng.Float64() - 0.5) * 2)
	}

	return weights, injection
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
