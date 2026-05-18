package kernels

import (
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestMatMulFloat32(t *testing.T) {
	convey.Convey("Given 2x3 × 3x2 float32 matrices", t, func() {
		leftShape, _ := tensor.NewShape([]int{2, 3})
		rightShape, _ := tensor.NewShape([]int{3, 2})
		outShape, _ := tensor.NewShape([]int{2, 2})

		left, _ := tensor.NewZeroed(leftShape, dtype.Float32)
		right, _ := tensor.NewZeroed(rightShape, dtype.Float32)
		out, _ := tensor.NewZeroed(outShape, dtype.Float32)

		leftView, _ := left.Float32Native()
		rightView, _ := right.Float32Native()

		for index, value := range []float32{1, 2, 3, 4, 5, 6} {
			leftView[index] = value
		}

		for index, value := range []float32{7, 8, 9, 10, 11, 12} {
			rightView[index] = value
		}

		kernel := Default.Lookup("matmul", Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		})

		convey.Convey("Output should be the matrix product", func() {
			err := kernel.Run(left, right, out)
			convey.So(err, convey.ShouldBeNil)

			outView, _ := out.Float32Native()
			// [[1 2 3]    [[7 8]     [[58 64]
			//  [4 5 6]] *  [9 10]  =  [139 154]]
			//              [11 12]]
			convey.So(outView, convey.ShouldResemble, []float32{58, 64, 139, 154})
		})
	})
}

func TestSoftmaxFloat32(t *testing.T) {
	convey.Convey("Given a 1x4 float32 tensor", t, func() {
		shape, _ := tensor.NewShape([]int{1, 4})
		input, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		inputView, _ := input.Float32Native()
		for index, value := range []float32{1, 2, 3, 4} {
			inputView[index] = value
		}

		kernel := Default.Lookup("softmax", Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		})

		convey.Convey("Output should sum to 1 and be monotone in input", func() {
			err := kernel.Run(input, out)
			convey.So(err, convey.ShouldBeNil)

			outView, _ := out.Float32Native()

			var sum float64
			for _, value := range outView {
				sum += float64(value)
			}

			convey.So(sum, convey.ShouldAlmostEqual, 1.0, 1e-5)
			convey.So(outView[0] < outView[1], convey.ShouldBeTrue)
			convey.So(outView[1] < outView[2], convey.ShouldBeTrue)
			convey.So(outView[2] < outView[3], convey.ShouldBeTrue)
		})
	})
}

func TestGELUFloat32(t *testing.T) {
	convey.Convey("Given a small float32 input", t, func() {
		shape, _ := tensor.NewShape([]int{3})
		input, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		inputView, _ := input.Float32Native()
		inputView[0] = -1
		inputView[1] = 0
		inputView[2] = 1

		kernel := Default.Lookup("gelu", Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		})

		convey.Convey("Output should match the exact erf-based GELU", func() {
			err := kernel.Run(input, out)
			convey.So(err, convey.ShouldBeNil)

			outView, _ := out.Float32Native()

			// GELU(0) is exactly 0, GELU(-1) ≈ -0.1587, GELU(1) ≈ 0.8413.
			convey.So(outView[1], convey.ShouldAlmostEqual, float32(0), 1e-6)
			convey.So(math.Abs(float64(outView[0])-(-0.1587)), convey.ShouldBeLessThan, 1e-3)
			convey.So(math.Abs(float64(outView[2])-0.8413), convey.ShouldBeLessThan, 1e-3)
		})
	})
}

func TestRMSNormFloat32(t *testing.T) {
	convey.Convey("Given an RMS-norm input", t, func() {
		shape, _ := tensor.NewShape([]int{1, 4})
		input, _ := tensor.NewZeroed(shape, dtype.Float32)
		scale, _ := tensor.NewZeroed(shape, dtype.Float32)
		out, _ := tensor.NewZeroed(shape, dtype.Float32)

		inputView, _ := input.Float32Native()
		scaleView, _ := scale.Float32Native()

		for index, value := range []float32{1, 2, 3, 4} {
			inputView[index] = value
			scaleView[index] = 1.0
		}

		kernel := Default.Lookup("rmsnorm", Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		})

		err := kernel.Run(input, scale, out)
		convey.So(err, convey.ShouldBeNil)

		outView, _ := out.Float32Native()

		convey.Convey("Mean square of the output should be ~1", func() {
			var meanSquare float64

			for _, value := range outView {
				meanSquare += float64(value) * float64(value)
			}

			meanSquare /= float64(len(outView))
			convey.So(meanSquare, convey.ShouldAlmostEqual, 1.0, 1e-4)
		})
	})
}
