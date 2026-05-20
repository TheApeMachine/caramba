package matmul

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func TestMatMulFloat16(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run(fmt.Sprintf("inner=%d", n), func(t *testing.T) {
			convey.Convey("Output should equal the float16 reference matmul", t, func() {
				left, right, out := newFloat16MatMulCase(t, n)
				err := RunMatMulFloat16(left, right, out)
				convey.So(err, convey.ShouldBeNil)

				assertFloat16MatMulCase(t, left, right, out, n)
			})
		})
	}
}

func newFloat16MatMulCase(
	testingObject testing.TB,
	inner int,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	leftShape, _ := tensor.NewShape([]int{2, inner})
	rightShape, _ := tensor.NewShape([]int{inner, 2})
	outShape, _ := tensor.NewShape([]int{2, 2})
	left, _ := tensor.NewZeroed(leftShape, dtype.Float16)
	right, _ := tensor.NewZeroed(rightShape, dtype.Float16)
	out, _ := tensor.NewZeroed(outShape, dtype.Float16)
	leftView, _ := left.Float16Native()
	rightView, _ := right.Float16Native()

	for index := range leftView {
		leftView[index] = dtype.Fromfloat32(float32((index%5)+1) * 0.25)
	}

	for index := range rightView {
		rightView[index] = dtype.Fromfloat32(float32((index%7)+1) * 0.5)
	}

	return left, right, out
}

func assertFloat16MatMulCase(
	testingObject testing.TB,
	left tensor.Tensor,
	right tensor.Tensor,
	out tensor.Tensor,
	inner int,
) {
	testingObject.Helper()

	leftView, _ := left.Float16Native()
	rightView, _ := right.Float16Native()
	outView, _ := out.Float16Native()

	for rowIndex := 0; rowIndex < 2; rowIndex++ {
		for colIndex := 0; colIndex < 2; colIndex++ {
			var sum float32

			for innerIndex := 0; innerIndex < inner; innerIndex++ {
				sum += leftView[rowIndex*inner+innerIndex].Float32() *
					rightView[innerIndex*2+colIndex].Float32()
			}

			convey.So(outView[rowIndex*2+colIndex], convey.ShouldEqual, dtype.Fromfloat32(sum))
		}
	}
}
