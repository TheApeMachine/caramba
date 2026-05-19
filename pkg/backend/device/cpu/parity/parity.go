package parity

import (
	"math"
	"testing"
)

/*
Lengths exercises edge alignment, single-vector, and multi-vector paths.
*/
var Lengths = []int{1, 7, 64, 1024, 8192}

/*
Float32ULPDistance returns the ULP gap between two float32 values.
*/
func Float32ULPDistance(left, right float32) int {
	leftBits := math.Float32bits(left)
	rightBits := math.Float32bits(right)

	if leftBits > rightBits {
		leftBits, rightBits = rightBits, leftBits
	}

	return int(rightBits - leftBits)
}

const nearZeroFloat32 = 1e-11

/*
AssertFloat32SlicesWithinULP fails when any lane exceeds maxULP from want.
*/
func AssertFloat32SlicesWithinULP(
	testingTB *testing.T,
	got, want []float32,
	maxULP int,
) {
	testingTB.Helper()

	if len(got) != len(want) {
		testingTB.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}

	for index := range got {
		if float32LanesMatch(got[index], want[index], maxULP) {
			continue
		}

		testingTB.Fatalf(
			"lane %d got=%g want=%g ulp=%d max=%d",
			index, got[index], want[index],
			Float32ULPDistance(got[index], want[index]), maxULP,
		)
	}
}

func float32LanesMatch(left, right float32, maxULP int) bool {
	if left == right {
		return true
	}

	if math.IsNaN(float64(left)) && math.IsNaN(float64(right)) {
		return true
	}

	leftAbs := math.Abs(float64(left))
	rightAbs := math.Abs(float64(right))

	if leftAbs < nearZeroFloat32 && rightAbs < nearZeroFloat32 {
		return true
	}

	return Float32ULPDistance(left, right) <= maxULP
}
