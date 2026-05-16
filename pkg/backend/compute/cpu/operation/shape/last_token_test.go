package shape

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestLastToken(t *testing.T) {
	Convey("Given a LastToken operation", t, func() {
		operation := NewLastToken()

		Convey("Forward", func() {
			Convey("It should select the final sequence row for each outer element", func() {
				outputState, err := operation.Forward(
					state.NewDict().
						WithShape([]int{2, 3, 2}).
						WithInput([]float64{
							1, 2,
							3, 4,
							5, 6,
							7, 8,
							9, 10,
							11, 12,
						}),
				)

				So(err, ShouldBeNil)
				So(outputState.Out, ShouldResemble, []float64{5, 6, 11, 12})
			})
		})
	})
}

func BenchmarkLastToken_Forward(benchmark *testing.B) {
	operation := NewLastToken()
	shape := []int{8, 512, 768}
	input := make([]float64, 8*512*768)

	for benchmark.Loop() {
		_, _ = operation.Forward(
			state.NewDict().
				WithShape(shape).
				WithInput(input),
		)
	}
}
