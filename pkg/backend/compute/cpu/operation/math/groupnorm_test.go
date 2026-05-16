package math

import (
	gomath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestGroupNorm_Forward(test *testing.T) {
	Convey("Given a GroupNorm operation", test, func() {
		operation := NewGroupNorm()

		Convey("Forward", func() {
			Convey("It should normalize NCHW tensors by group", func() {
				input := []float64{1, 2, 3, 4}
				stateDict := state.NewDict().
					WithShape([]int{1, 2, 1, 2}).
					WithInput(input).
					WithWeight([]float64{1, 1}).
					WithBias([]float64{0, 0}).
					WithEps(0)
				stateDict.Groups = 1

				output, err := operation.Forward(stateDict)

				So(err, ShouldBeNil)
				assertGroupNormClose(output.Out, []float64{
					-1.3416407864998738,
					-0.4472135954999579,
					0.4472135954999579,
					1.3416407864998738,
				})
			})

			Convey("It should apply per-channel affine parameters", func() {
				input := []float64{1, 2, 3, 4}
				stateDict := state.NewDict().
					WithShape([]int{1, 2, 1, 2}).
					WithInput(input).
					WithWeight([]float64{2, 3}).
					WithBias([]float64{10, 20}).
					WithEps(0)
				stateDict.Groups = 2

				output, err := operation.Forward(stateDict)

				So(err, ShouldBeNil)
				assertGroupNormClose(output.Out, []float64{8, 12, 17, 23})
			})

			Convey("It should reject non-divisible channel groups", func() {
				stateDict := state.NewDict().
					WithShape([]int{1, 3, 1, 1}).
					WithInput([]float64{1, 2, 3}).
					WithWeight([]float64{1, 1, 1}).
					WithBias([]float64{0, 0, 0})
				stateDict.Groups = 2

				_, err := operation.Forward(stateDict)

				So(err, ShouldNotBeNil)
				So(err.Error(), ShouldContainSubstring, "channels 3 must be divisible by groups 2")
			})
		})
	})
}

func BenchmarkGroupNorm_Forward(benchmark *testing.B) {
	operation := NewGroupNorm()
	input := mathSequence(2*32*16*16, 0.01, -1.0)
	weight := make([]float64, 32)
	bias := make([]float64, 32)

	for index := range weight {
		weight[index] = 1
	}

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{2, 32, 16, 16}).
			WithInput(input).
			WithWeight(weight).
			WithBias(bias).
			WithEps(1e-5)
		stateDict.Groups = 32
		_, _ = operation.Forward(stateDict)
	}
}

func assertGroupNormClose(actual, expected []float64) {
	So(len(actual), ShouldEqual, len(expected))

	for index := range expected {
		So(gomath.Abs(actual[index]-expected[index]), ShouldBeLessThan, 1e-9)
	}
}
