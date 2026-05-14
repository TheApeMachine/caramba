package math

import (
	gomath "math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/state"
)

func TestExp_Forward(t *testing.T) {
	Convey("Given an Exp operation", t, func() {
		op := NewExp()

		Convey("Forward", func() {
			Convey("It should match math.Exp for sample inputs", func() {
				inputs := []float64{0, 1, -1, 0.5, -0.5, 2.5, -2.5, 5.0, -5.0, 10.0, -10.0, 100.0, -100.0}
				out := forwardMath(op, []int{len(inputs)}, inputs)
				for index, value := range inputs {
					expected := gomath.Exp(value)
					relErr := gomath.Abs(out[index]-expected) / gomath.Abs(expected)
					So(relErr, ShouldBeLessThan, 1e-7)
				}
			})

			Convey("It should handle very large and very small inputs by clamping", func() {
				out := forwardMath(op, []int{2}, []float64{1000.0, -1000.0})
				So(gomath.IsInf(out[0], 0) || out[0] > 1e300, ShouldBeTrue)
				So(out[1], ShouldBeLessThan, 1e-300)
			})

			Convey("It should match math.Exp for a long random input across SIMD lanes", func() {
				inputs := make([]float64, 257)
				for index := range inputs {
					inputs[index] = float64(index)*0.1 - 12.8
				}
				out := forwardMath(op, []int{len(inputs)}, inputs)
				for index, value := range inputs {
					expected := gomath.Exp(value)
					relErr := gomath.Abs(out[index]-expected) / gomath.Abs(expected)
					So(relErr, ShouldBeLessThan, 1e-7)
				}
			})
		})
	})
}

func BenchmarkExp_Forward(b *testing.B) {
	op := NewExp()
	data := make([]float64, 1024)
	for index := range data {
		data[index] = float64(index)*0.01 - 5.0
	}
	for b.Loop() {
		stateDict := state.NewDict().
			WithShape([]int{1024}).
			WithInput(data)
		_, _ = op.Forward(stateDict)
	}
}
