package math

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestSign_Forward(t *testing.T) {
	Convey("Given a Sign operation", t, func() {
		op := NewSign()

		Convey("Forward", func() {
			Convey("It should return +1 for positive, -1 for negative, 0 for zero", func() {
				out := op.Forward([]int{5}, []float64{3.0, -2.5, 0.0, 0.001, -100.0})
				So(out, ShouldResemble, []float64{1, -1, 0, 1, -1})
			})

			Convey("It should handle an all-positive input", func() {
				out := op.Forward([]int{3}, []float64{1.0, 2.0, 3.0})
				So(out, ShouldResemble, []float64{1, 1, 1})
			})

			Convey("It should handle an all-negative input", func() {
				out := op.Forward([]int{3}, []float64{-1.0, -2.0, -3.0})
				So(out, ShouldResemble, []float64{-1, -1, -1})
			})
		})
	})
}

func BenchmarkSign_Forward(b *testing.B) {
	op := NewSign()
	data := make([]float64, 1024)

	for idx := range data {
		data[idx] = float64(idx%3 - 1)
	}

	b.ResetTimer()

	for repeat := 0; repeat < b.N; repeat++ {
		op.Forward([]int{1024}, data)
	}
}
