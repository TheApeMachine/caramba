package shape

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestUpsampleNearest2D(t *testing.T) {
	Convey("Given an UpsampleNearest2D operation", t, func() {
		operation := NewUpsampleNearest2D()

		Convey("Forward", func() {
			Convey("It should expand an NCHW tensor by nearest-neighbor scale", func() {
				stateDict := state.NewDict().
					WithShape([]int{1, 1, 2, 2}).
					WithInput([]float64{1, 2, 3, 4})
				stateDict.ScaleH = 2
				stateDict.ScaleW = 2

				outputState, err := operation.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, []float64{
					1, 1, 2, 2,
					1, 1, 2, 2,
					3, 3, 4, 4,
					3, 3, 4, 4,
				})
			})

			Convey("It should derive scale from explicit output dimensions", func() {
				stateDict := state.NewDict().
					WithShape([]int{1, 2, 1, 2}).
					WithInput([]float64{1, 2, 3, 4})
				stateDict.OutH = 3
				stateDict.OutW = 4

				outputState, err := operation.Forward(stateDict)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, []float64{
					1, 1, 2, 2,
					1, 1, 2, 2,
					1, 1, 2, 2,
					3, 3, 4, 4,
					3, 3, 4, 4,
					3, 3, 4, 4,
				})
			})

			Convey("It should reject non-NCHW input", func() {
				stateDict := state.NewDict().
					WithShape([]int{1, 2, 2}).
					WithInput([]float64{1, 2, 3, 4})
				stateDict.ScaleH = 2
				stateDict.ScaleW = 2

				outputState, err := operation.Forward(stateDict)

				So(err, ShouldNotBeNil)
				So(outputState, ShouldBeNil)
			})
		})
	})
}

func BenchmarkUpsampleNearest2D_Forward(benchmark *testing.B) {
	operation := NewUpsampleNearest2D()
	shape := []int{1, 32, 128, 128}
	input := make([]float64, 1*32*128*128)

	for index := range input {
		input[index] = float64(index)
	}

	for benchmark.Loop() {
		stateDict := state.NewDict().
			WithShape(shape).
			WithInput(input)
		stateDict.ScaleH = 2
		stateDict.ScaleW = 2
		_, _ = operation.Forward(stateDict)
	}
}
