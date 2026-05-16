package math

import (
	gomath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestSoftmax_Forward(t *testing.T) {
	Convey("Given a Softmax operation", t, func() {
		operation := NewSoftmax()

		Convey("Forward", func() {
			Convey("It should match a stable reference across SIMD row sizes", func() {
				for _, length := range []int{1, 7, 64, 1024, 8192} {
					input := mathSequence(length, 0.03, -1.2)
					output := forwardMath(operation, []int{length}, input)
					expected := referenceSoftmax(input)
					maxError := 0.0
					sum := 0.0

					for index := range expected {
						errorValue := gomath.Abs(output[index] - expected[index])

						if errorValue > maxError {
							maxError = errorValue
						}

						sum += output[index]
					}

					So(maxError, ShouldBeLessThan, 1e-7)
					So(sum, ShouldAlmostEqual, 1.0, 1e-9)
				}
			})
		})
	})
}

func BenchmarkSoftmax_Forward(benchmark *testing.B) {
	operation := NewSoftmax()
	input := mathSequence(8192, 0.03, -1.2)

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{8192}).
			WithInput(input)
		_, _ = operation.Forward(stateDict)
	}
}

func referenceSoftmax(input []float64) []float64 {
	output := make([]float64, len(input))
	maxValue := input[0]

	for _, value := range input[1:] {
		if value > maxValue {
			maxValue = value
		}
	}

	sum := 0.0

	for index, value := range input {
		output[index] = gomath.Exp(value - maxValue)
		sum += output[index]
	}

	for index := range output {
		output[index] /= sum
	}

	return output
}
