package math

import (
	gomath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestLogSumExp_Forward(t *testing.T) {
	Convey("Given a LogSumExp operation", t, func() {
		operation := NewLogSumExp()

		Convey("Forward", func() {
			Convey("It should match a stable reference across SIMD row sizes", func() {
				for _, length := range []int{1, 7, 64, 1024, 8192} {
					input := mathSequence(length, 0.025, -0.7)
					output := forwardMath(operation, []int{length}, input)
					expected := referenceLogSumExp(input)

					So(output, ShouldHaveLength, 1)
					So(output[0], ShouldAlmostEqual, expected, 1e-7)
				}
			})
		})
	})
}

func BenchmarkLogSumExp_Forward(benchmark *testing.B) {
	operation := NewLogSumExp()
	input := mathSequence(8192, 0.025, -0.7)

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{8192}).
			WithInput(input)
		_, _ = operation.Forward(stateDict)
	}
}

func referenceLogSumExp(input []float64) float64 {
	maxValue := input[0]

	for _, value := range input[1:] {
		if value > maxValue {
			maxValue = value
		}
	}

	sum := 0.0

	for _, value := range input {
		sum += gomath.Exp(value - maxValue)
	}

	return maxValue + gomath.Log(sum)
}
