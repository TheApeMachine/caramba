package shape

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestTranspose(t *testing.T) {
	Convey("Given a Transpose operation swapping dims 0 and 1", t, func() {
		op := NewTranspose(0, 1)

		Convey("Forward", func() {
			Convey("It should transpose a 2x3 matrix to 3x2", func() {
				// [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
				in := []float64{1, 2, 3, 4, 5, 6}
				out := op.Forward([]int{2, 3}, in)
				So(out, ShouldResemble, []float64{1, 4, 2, 5, 3, 6})
			})

			Convey("It should be its own inverse", func() {
				in := []float64{1, 2, 3, 4, 5, 6, 7, 8}
				shape := []int{2, 4}
				out := op.Forward(shape, in)
				back := op.Forward([]int{4, 2}, out)
				So(back, ShouldResemble, in)
			})

			Convey("It should handle square matrices", func() {
				in := []float64{1, 2, 3, 4}
				out := op.Forward([]int{2, 2}, in)
				So(out, ShouldResemble, []float64{1, 3, 2, 4})
			})
		})
	})
}

func BenchmarkTranspose_Forward(b *testing.B) {
	op := NewTranspose(0, 1)
	rows, cols := 512, 512
	in := make([]float64, rows*cols)
	for i := range in {
		in[i] = float64(i)
	}
	shape := []int{rows, cols}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward(shape, in)
	}
}
