package math

import (
	"fmt"
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

			Convey("It should handle single-element vectors", func() {
				out := op.Forward([]int{1, 1}, []float64{3}, []float64{4})
				So(out, ShouldResemble, []float64{12})
			})

			Convey("It should handle a row containing zeros", func() {
				out := op.Forward([]int{2, 2}, []float64{0, 1}, []float64{5, 7})
				So(out, ShouldResemble, []float64{0, 0, 5, 7})
			})

			Convey("It should return empty for zero dimensions", func() {
				out := op.Forward([]int{0, 0}, []float64{}, []float64{})
				So(len(out), ShouldEqual, 0)
			})
		})
	})
}

func BenchmarkOuter_Forward(b *testing.B) {
	op := NewOuter()
	// sizes to cover SIMD stride boundaries
	sizes := []int{32, 64, 128, 256, 512}

	for _, N := range sizes {
		N := N
		b.Run(fmt.Sprintf("N=%d", N), func(b *testing.B) {
			vecA := make([]float64, N)
			vecB := make([]float64, N)
			for idx := range N {
				vecA[idx] = float64(idx%2*2 - 1)
				vecB[idx] = float64(idx%2*2 - 1)
			}
			b.ResetTimer()
			for range b.N {
				op.Forward([]int{N, N}, vecA, vecB)
			}
		})
	}
}
