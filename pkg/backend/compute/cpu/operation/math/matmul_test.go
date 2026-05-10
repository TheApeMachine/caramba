package math

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMatmul(t *testing.T) {
	Convey("Given a Matmul operation", t, func() {
		op := NewMatmul()

		Convey("Forward", func() {
			Convey("It should multiply 2x2 identity matrices correctly", func() {
				// I @ I = I
				eye := []float64{1, 0, 0, 1}
				out := op.Forward([]int{2, 2, 2}, eye, eye)
				So(out, ShouldResemble, []float64{1, 0, 0, 1})
			})

			Convey("It should multiply a row vector by a column vector", func() {
				// [1,2,3,4] @ [1,2,3,4]^T as [1x4] @ [4x1] = [30]
				a := []float64{1, 2, 3, 4}
				b := []float64{1, 2, 3, 4}
				out := op.Forward([]int{1, 4, 1}, a, b)
				So(out, ShouldHaveLength, 1)
				So(out[0], ShouldAlmostEqual, 30.0, 1e-9)
			})

			Convey("It should produce correct output shape for MxK times KxN", func() {
				// 3x4 @ 4x2 = 3x2
				a := make([]float64, 12)
				b := make([]float64, 8)
				out := op.Forward([]int{3, 4, 2}, a, b)
				So(out, ShouldHaveLength, 6)
			})
		})
	})
}

func BenchmarkMatmul_Forward(b *testing.B) {
	op := NewMatmul()
	m, k, n := 64, 64, 64
	a := make([]float64, m*k)
	mat := make([]float64, k*n)
	for i := range a {
		a[i] = float64(i) / float64(m*k)
	}
	for i := range mat {
		mat[i] = float64(i) / float64(k*n)
	}
	shape := []int{m, k, n}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		op.Forward(shape, a, mat)
	}
}
