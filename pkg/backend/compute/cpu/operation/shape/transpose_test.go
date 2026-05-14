package shape

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestTranspose(t *testing.T) {
	Convey("Given a Transpose operation swapping dims 0 and 1", t, func() {
		op := NewTranspose()
		swap01 := func(stateDict *state.Dict) {
			stateDict.Dim0 = 0
			stateDict.Dim1 = 1
		}

		Convey("Forward", func() {
			Convey("It should transpose a 2x3 matrix to 3x2", func() {
				in := []float64{1, 2, 3, 4, 5, 6}
				out := forwardShape(op, []int{2, 3}, in, swap01)
				So(out, ShouldResemble, []float64{1, 4, 2, 5, 3, 6})
			})

			Convey("It should be its own inverse", func() {
				in := []float64{1, 2, 3, 4, 5, 6, 7, 8}
				shape := []int{2, 4}
				out := forwardShape(op, shape, in, swap01)
				back := forwardShape(op, []int{4, 2}, out, swap01)
				So(back, ShouldResemble, in)
			})

			Convey("It should handle square matrices", func() {
				in := []float64{1, 2, 3, 4}
				out := forwardShape(op, []int{2, 2}, in, swap01)
				So(out, ShouldResemble, []float64{1, 3, 2, 4})
			})
		})
	})
}

func BenchmarkTranspose_Forward(b *testing.B) {
	op := NewTranspose()
	rows, cols := 512, 512
	in := make([]float64, rows*cols)
	for index := range in {
		in[index] = float64(index)
	}
	shape := []int{rows, cols}

	for b.Loop() {
		stateDict := state.NewDict().
			WithShape(shape).
			WithInput(in)
		stateDict.Dim0 = 0
		stateDict.Dim1 = 1
		_, _ = op.Forward(stateDict)
	}
}
