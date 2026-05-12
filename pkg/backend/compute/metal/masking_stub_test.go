//go:build !darwin || !cgo

package metal

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMetalCausalMask_Forward(t *testing.T) {
	Convey("Given an unavailable Metal causal mask", t, func() {
		op := &MetalCausalMask{}

		Convey("It should return the backend error instead of a scalar fallback", func() {
			out, err := op.Forward([]int{2})

			So(err, ShouldNotBeNil)
			So(err, ShouldEqual, errMetalUnavailable)
			So(out, ShouldBeNil)
		})
	})
}

func TestMetalApplyMask_Forward(t *testing.T) {
	Convey("Given an unavailable Metal apply mask", t, func() {
		op := &MetalApplyMask{}

		Convey("It should return the backend error instead of adding on CPU", func() {
			out, err := op.Forward([]int{2}, []float64{1, 2}, []float64{3, 4})

			So(err, ShouldNotBeNil)
			So(err, ShouldEqual, errMetalUnavailable)
			So(out, ShouldBeNil)
		})
	})
}
