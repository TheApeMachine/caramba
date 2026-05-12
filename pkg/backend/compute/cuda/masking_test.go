//go:build linux && cgo && cuda

package cuda

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestCUDACausalMaskForward(t *testing.T) {
	Convey("Given a CUDA causal mask operation", t, func() {
		operation := NewMasking().NewCausalMask()

		Convey("It should return errors instead of fabricating fallback results", func() {
			output, err := operation.Forward(nil)

			So(err, ShouldNotBeNil)
			So(output, ShouldBeNil)
		})
	})
}

func TestCUDAApplyMaskForward(t *testing.T) {
	Convey("Given a CUDA apply mask operation", t, func() {
		operation := NewMasking().NewApplyMask()

		Convey("It should reject missing inputs instead of fabricating fallback results", func() {
			output, err := operation.Forward([]int{2}, []float64{1, 2})

			So(err, ShouldNotBeNil)
			So(output, ShouldBeNil)
		})

		Convey("It should reject mismatched input lengths before dispatch", func() {
			output, err := operation.Forward(
				[]int{2},
				[]float64{1, 2},
				[]float64{3},
			)

			So(err, ShouldNotBeNil)
			So(output, ShouldBeNil)
		})
	})
}

func BenchmarkCUDAApplyMaskForwardValidation(b *testing.B) {
	operation := NewMasking().NewApplyMask()

	for b.Loop() {
		_, _ = operation.Forward([]int{2}, []float64{1, 2})
	}
}
