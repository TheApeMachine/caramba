package neon

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

var parityNs = []int{1, 7, 64, 1024, 8192}

func TestAddFloat32(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			convey.Convey("Given two float32 host tensors", t, func() {
				kernel, ok := Default.Lookup("add", Signature{
					Layout:  tensor.LayoutDense,
					Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
					Outputs: []dtype.DType{dtype.Float32},
				})
				convey.So(ok, convey.ShouldBeTrue)

				shape, _ := tensor.NewShape([]int{n})
				left, _ := tensor.NewZeroed(shape, dtype.Float32)
				right, _ := tensor.NewZeroed(shape, dtype.Float32)
				out, _ := tensor.NewZeroed(shape, dtype.Float32)

				leftView, _ := left.Float32Native()
				rightView, _ := right.Float32Native()

				for index := range leftView {
					leftView[index] = float32(index + 1)
					rightView[index] = float32(10 * (index + 1))
				}

				err := kernel.Run(left, right, out)
				convey.So(err, convey.ShouldBeNil)

				outView, _ := out.Float32Native()

				for index := range outView {
					convey.So(outView[index], convey.ShouldEqual, leftView[index]+rightView[index])
				}
			})
		})
	}
}

func TestAddBFloat16_MixedAccumulation(t *testing.T) {
	convey.Convey("Given two bf16 tensors", t, func() {
		shape, _ := tensor.NewShape([]int{3})
		left, _ := tensor.NewZeroed(shape, dtype.BFloat16)
		right, _ := tensor.NewZeroed(shape, dtype.BFloat16)
		out, _ := tensor.NewZeroed(shape, dtype.BFloat16)

		leftView, _ := left.BFloat16Native()
		rightView, _ := right.BFloat16Native()

		originals := []float32{1.0, -2.0, 0.5}

		for index, value := range originals {
			leftView[index] = dtype.NewBfloat16FromFloat32(value)
			rightView[index] = dtype.NewBfloat16FromFloat32(value * 2)
		}

		kernel, ok := Default.Lookup("add", Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		})
		convey.So(ok, convey.ShouldBeTrue)

		err := kernel.Run(left, right, out)

		convey.Convey("Output should equal the elementwise sum within BF16 ULP", func() {
			convey.So(err, convey.ShouldBeNil)

			outView, _ := out.BFloat16Native()

			for index, source := range originals {
				expected := source + source*2
				actual := (&outView[index]).Float32()

				convey.So(actual, convey.ShouldAlmostEqual, expected, 1e-2)
			}
		})
	})
}
