package vsa

import (
	"math"
	"math/rand"
)

const vsaScalarMaxULP = 0

func randomVSAVectors(
	count int,
	seed int64,
) (first, second []float32) {
	randomSource := rand.New(rand.NewSource(seed))

	first = make([]float32, count)
	second = make([]float32, count)

	for index := range first {
		first[index] = randomSource.Float32()*2 - 1
		second[index] = randomSource.Float32()*2 - 1
	}

	return first, second
}

func vsaParityShift(length int) int {
	if length <= 1 {
		return 0
	}

	return 1 + (length % min(length-1, 7))
}

func assertVSASliceParity(
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
		if diff > float64(vsaScalarMaxULP) {
			testingTB.Fatalf(
				"index %d got=%g want=%g diff=%g",
				index, got[index], want[index], got[index]-want[index],
			)
		}
	}
}

func assertVSASimilarityParity(
	testingTB interface {
		Helper()
		Fatalf(string, ...any)
	},
	got, want float32,
) {
	testingTB.Helper()

	diff := math.Abs(float64(got - want))
	if diff > float64(vsaScalarMaxULP) {
		testingTB.Fatalf("got=%g want=%g diff=%g", got, want, got-want)
	}
}
