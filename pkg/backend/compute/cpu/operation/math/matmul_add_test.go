package math

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestMatmulAdd_Forward(t *testing.T) {
	Convey("Given a MatmulAdd operation", t, func() {
		operation := NewMatmulAdd()

		Convey("It should multiply matrices and broadcast column bias", func() {
			stateDict := state.NewDict().
				WithShape([]int{2, 3, 2}).
				WithInputs(
					[]float64{
						1, 2, 3,
						4, 5, 6,
					},
					[]float64{
						7, 8,
						9, 10,
						11, 12,
					},
					[]float64{1, -1},
				)

			output, err := operation.Forward(stateDict)

			So(err, ShouldBeNil)
			So(output.Out, ShouldResemble, []float64{59, 63, 140, 153})
		})

		Convey("It should accept a full output bias", func() {
			stateDict := state.NewDict().
				WithShape([]int{1, 2, 2}).
				WithInputs(
					[]float64{1, -1},
					[]float64{1, 2, 3, 4},
					[]float64{10, 20},
				)

			output, err := operation.Forward(stateDict)

			So(err, ShouldBeNil)
			So(output.Out, ShouldResemble, []float64{8, 18})
		})
	})
}

func BenchmarkMatmulAdd_Forward(benchmark *testing.B) {
	operation := NewMatmulAdd()
	left := make([]float64, 64*64)
	right := make([]float64, 64*64)
	bias := make([]float64, 64)

	for index := range left {
		left[index] = float64(index%17) * 0.125
		right[index] = float64(index%19) * 0.0625
	}

	for index := range bias {
		bias[index] = float64(index) * 0.01
	}

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{64, 64, 64}).
			WithInputs(left, right, bias)

		if _, err := operation.Forward(stateDict); err != nil {
			benchmark.Fatal(err)
		}
	}
}
