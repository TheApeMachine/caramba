package predictive_coding

import (
	"math"
	"math/rand"
)

const (
	predictiveCodingScalarMaxULP = 0
	predictiveCodingTestLR       = 1e-2
)

func randomPredictiveCodingVectors(
	count int,
	seed int64,
) (first, second, third []float32) {
	randomSource := rand.New(rand.NewSource(seed))

	first = make([]float32, count)
	second = make([]float32, count)
	third = make([]float32, count)

	for index := range first {
		first[index] = randomSource.Float32()*2 - 1
		second[index] = randomSource.Float32()*2 - 1
		third[index] = randomSource.Float32()*2 - 1
	}

	return first, second, third
}

func randomPredictiveCodingWeights(
	outDim, inDim int,
	seed int64,
) []float32 {
	randomSource := rand.New(rand.NewSource(seed))
	weights := make([]float32, outDim*inDim)

	for index := range weights {
		weights[index] = randomSource.Float32()*0.5 - 0.25
	}

	return weights
}

func predictiveCodingParityDims(length int) (outDim, inDim int) {
	return length, length
}

func assertPredictionParity(
	testingTB interface {
		Helper()
		Fatalf(string, ...any)
	},
	got, want []float32,
	outDim int,
) {
	testingTB.Helper()

	if len(got) != outDim || len(want) != outDim {
		testingTB.Fatalf("length mismatch got=%d want=%d outDim=%d", len(got), len(want), outDim)
	}

	for index := range got {
		if got[index] != want[index] {
			testingTB.Fatalf(
				"index %d got=%g want=%g diff=%g",
				index, got[index], want[index], got[index]-want[index],
			)
		}
	}
}

func assertSliceParity(
	testingTB interface {
		Helper()
		Fatalf(string, ...any)
	},
	got, want []float32,
) {
	testingTB.Helper()

	if len(got) != len(want) {
		testingTB.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}

	for index := range got {
		diff := math.Abs(float64(got[index] - want[index]))
		if diff > float64(predictiveCodingScalarMaxULP) {
			testingTB.Fatalf(
				"index %d got=%g want=%g diff=%g",
				index, got[index], want[index], got[index]-want[index],
			)
		}
	}
}
