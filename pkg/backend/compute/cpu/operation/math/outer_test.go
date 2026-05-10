package math

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestOuter_Forward(t *testing.T) {
	Convey("Given an Outer operation", t, func() {
		op := NewOuter()

		Convey("Forward", func() {
			Convey("It should compute the outer product of two vectors", func() {
				// a=[1,2], b=[3,4,5] → [[3,4,5],[6,8,10]]
				out := op.Forward([]int{2, 3}, []float64{1, 2}, []float64{3, 4, 5})
				So(out, ShouldResemble, []float64{3, 4, 5, 6, 8, 10})
			})

			Convey("It should produce a symmetric matrix for identical inputs", func() {
				// a=b=[1,2] → [[1,2],[2,4]] (Hebbian W for one pattern)
				out := op.Forward([]int{2, 2}, []float64{1, 2}, []float64{1, 2})
				So(out, ShouldResemble, []float64{1, 2, 2, 4})
			})

			Convey("It should handle bipolar patterns as used in Hopfield networks", func() {
				// pattern=[1,-1,1], W += outer(p,p)
				out := op.Forward([]int{3, 3}, []float64{1, -1, 1}, []float64{1, -1, 1})
				So(out, ShouldResemble, []float64{1, -1, 1, -1, 1, -1, 1, -1, 1})
			})
		})
	})
}

func BenchmarkOuter_Forward(b *testing.B) {
	op := NewOuter()
	N := 128
	a := make([]float64, N)
	bb := make([]float64, N)

	for idx := range N {
		a[idx] = float64(idx%2*2 - 1)
		bb[idx] = float64(idx%2*2 - 1)
	}

	b.ResetTimer()

	for repeat := 0; repeat < b.N; repeat++ {
		op.Forward([]int{N, N}, a, bb)
	}
}
