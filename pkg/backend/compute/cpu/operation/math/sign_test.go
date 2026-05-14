package math

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestSign_Forward(t *testing.T) {
	Convey("Given a Sign operation", t, func() {
		op := NewSign()

		Convey("Forward", func() {
			Convey("It should return +1 for positive, -1 for negative, 0 for zero", func() {
				out := forwardMath(op, []int{5}, []float64{3.0, -2.5, 0.0, 0.001, -100.0})
				So(out, ShouldResemble, []float64{1, -1, 0, 1, -1})
			})

			Convey("It should handle an all-positive input", func() {
				out := forwardMath(op, []int{3}, []float64{1.0, 2.0, 3.0})
				So(out, ShouldResemble, []float64{1, 1, 1})
			})

			Convey("It should handle an all-negative input", func() {
				out := forwardMath(op, []int{3}, []float64{-1.0, -2.0, -3.0})
				So(out, ShouldResemble, []float64{-1, -1, -1})
			})

			Convey("It should return an empty output for an empty input", func() {
				out := forwardMath(op, []int{0}, []float64{})
				So(out, ShouldBeEmpty)
			})

			Convey("It should reject missing inputs", func() {
				_, err := op.Forward(state.NewDict().WithShape([]int{3}))

				So(err, ShouldNotBeNil)
			})
		})
	})
}

func BenchmarkSign_Forward(b *testing.B) {
	op := NewSign()
	data := make([]float64, 1024)

	for index := range data {
		data[index] = float64(index%3 - 1)
	}

	for b.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{1024}).
			WithInput(data)
		_, _ = op.Forward(stateDict)
	}
}
