package neon

import (
	"fmt"
	"math"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestMatMulFloat32(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run(fmt.Sprintf("inner=%d", n), func(t *testing.T) {
			convey.Convey("Output should equal the reference matmul", t, func() {
				kernel, ok := Default.Lookup("matmul", Signature{
					Layout:  tensor.LayoutDense,
					Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
					Outputs: []dtype.DType{dtype.Float32},
				})
				convey.So(ok, convey.ShouldBeTrue)

				leftShape, _ := tensor.NewShape([]int{2, n})
				rightShape, _ := tensor.NewShape([]int{n, 2})
				outShape, _ := tensor.NewShape([]int{2, 2})

				left, _ := tensor.NewZeroed(leftShape, dtype.Float32)
				right, _ := tensor.NewZeroed(rightShape, dtype.Float32)
				out, _ := tensor.NewZeroed(outShape, dtype.Float32)

				leftView, _ := left.Float32Native()
				rightView, _ := right.Float32Native()

				for index := range leftView {
					leftView[index] = float32((index%5)+1) * 0.25
				}

				for index := range rightView {
					rightView[index] = float32((index%7)+1) * 0.5
				}

				err := kernel.Run(left, right, out)
				convey.So(err, convey.ShouldBeNil)

				outView, _ := out.Float32Native()

				expected := make([]float32, 4)

				for rowIndex := 0; rowIndex < 2; rowIndex++ {
					for colIndex := 0; colIndex < 2; colIndex++ {
						var sum float32

						for innerIndex := 0; innerIndex < n; innerIndex++ {
							sum += leftView[rowIndex*n+innerIndex] *
								rightView[innerIndex*2+colIndex]
						}

						expected[rowIndex*2+colIndex] = sum
					}
				}

				for index, want := range expected {
					convey.So(outView[index], convey.ShouldAlmostEqual, want, math.Max(1e-3, float64(want)*1e-4))
				}
			})
		})
	}
}

func TestSoftmaxFloat32(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			convey.Convey("Output should sum to 1 and be monotone", t, func() {
				kernel, ok := Default.Lookup("softmax", Signature{
					Layout:  tensor.LayoutDense,
					Inputs:  []dtype.DType{dtype.Float32},
					Outputs: []dtype.DType{dtype.Float32},
				})
				convey.So(ok, convey.ShouldBeTrue)

				shape, _ := tensor.NewShape([]int{1, n})
				input, _ := tensor.NewZeroed(shape, dtype.Float32)
				out, _ := tensor.NewZeroed(shape, dtype.Float32)

				inputView, _ := input.Float32Native()

				for index := range inputView {
					inputView[index] = float32(index + 1)
				}

				err := kernel.Run(input, out)
				convey.So(err, convey.ShouldBeNil)

				outView, _ := out.Float32Native()

				var sum float64

				for _, value := range outView {
					sum += float64(value)
				}

				convey.So(sum, convey.ShouldAlmostEqual, 1.0, 1e-4)

				monotone := true

				for index := 1; index < len(outView); index++ {
					if outView[index-1] > outView[index] {
						monotone = false
						break
					}
				}

				convey.So(monotone, convey.ShouldBeTrue)
			})
		})
	}
}

func TestGELUFloat32(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			convey.Convey("Output should match the exact erf-based GELU", t, func() {
				kernel, ok := Default.Lookup("gelu", Signature{
					Layout:  tensor.LayoutDense,
					Inputs:  []dtype.DType{dtype.Float32},
					Outputs: []dtype.DType{dtype.Float32},
				})
				convey.So(ok, convey.ShouldBeTrue)

				shape, _ := tensor.NewShape([]int{n})
				input, _ := tensor.NewZeroed(shape, dtype.Float32)
				out, _ := tensor.NewZeroed(shape, dtype.Float32)

				inputView, _ := input.Float32Native()

				for index := range inputView {
					inputView[index] = float32(index%7) - 3
				}

				err := kernel.Run(input, out)
				convey.So(err, convey.ShouldBeNil)

				outView, _ := out.Float32Native()

				const sqrtTwo = 1.41421356237309504880

				for index, value := range inputView {
					expected := 0.5 * value * float32(1+math.Erf(float64(value)/sqrtTwo))
					convey.So(math.Abs(float64(outView[index]-expected)), convey.ShouldBeLessThan, 1e-3)
				}
			})
		})
	}
}

func TestRMSNormFloat32(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			convey.Convey("Mean square of the output should be ~1", t, func() {
				kernel, ok := Default.Lookup("rmsnorm", Signature{
					Layout:  tensor.LayoutDense,
					Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
					Outputs: []dtype.DType{dtype.Float32},
				})
				convey.So(ok, convey.ShouldBeTrue)

				shape, _ := tensor.NewShape([]int{1, n})
				input, _ := tensor.NewZeroed(shape, dtype.Float32)
				scaleShape, _ := tensor.NewShape([]int{n})
				scale, _ := tensor.NewZeroed(scaleShape, dtype.Float32)
				out, _ := tensor.NewZeroed(shape, dtype.Float32)

				inputView, _ := input.Float32Native()
				scaleView, _ := scale.Float32Native()

				for index := range inputView {
					inputView[index] = float32(index%11) + 1
					scaleView[index] = 1.0
				}

				err := kernel.Run(input, scale, out)
				convey.So(err, convey.ShouldBeNil)

				outView, _ := out.Float32Native()

				var meanSquare float64

				for _, value := range outView {
					meanSquare += float64(value) * float64(value)
				}

				meanSquare /= float64(len(outView))
				convey.So(meanSquare, convey.ShouldAlmostEqual, 1.0, 1e-3)
			})
		})
	}
}
