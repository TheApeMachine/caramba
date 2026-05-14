package math

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestMatmul(t *testing.T) {
	Convey("Given a Matmul operation", t, func() {
		op := NewMatmul()

		Convey("Forward", func() {
			Convey("It should multiply 2x2 identity matrices correctly", func() {
				eye := []float64{1, 0, 0, 1}
				out := forwardMath(op, []int{2, 2, 2}, eye, eye)
				So(out, ShouldResemble, []float64{1, 0, 0, 1})
			})

			Convey("It should multiply a row vector by a column vector", func() {
				a := []float64{1, 2, 3, 4}
				b := []float64{1, 2, 3, 4}
				out := forwardMath(op, []int{1, 4, 1}, a, b)
				So(out, ShouldHaveLength, 1)
				So(out[0], ShouldAlmostEqual, 30.0, 1e-9)
			})

			Convey("It should produce correct output shape for MxK times KxN", func() {
				a := make([]float64, 12)
				b := make([]float64, 8)
				out := forwardMath(op, []int{3, 4, 2}, a, b)
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
	for index := range a {
		a[index] = float64(index) / float64(m*k)
	}
	for index := range mat {
		mat[index] = float64(index) / float64(k*n)
	}
	shape := []int{m, k, n}

	for b.Loop() {
		stateDict := state.NewDict().
			WithShape(shape).
			WithInputs(a, mat)
		_, _ = op.Forward(stateDict)
	}
}
