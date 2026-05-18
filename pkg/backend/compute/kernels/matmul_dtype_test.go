package kernels

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func TestMatMulFloat16(t *testing.T) {
	for _, n := range parityNs {
		n := n

		t.Run(fmt.Sprintf("inner=%d", n), func(t *testing.T) {
			convey.Convey("Output should equal the float16 reference matmul", t, func() {
				kernel, ok := Default.Lookup("matmul", Signature{
					Layout:  tensor.LayoutDense,
					Inputs:  []dtype.DType{dtype.Float16, dtype.Float16},
					Outputs: []dtype.DType{dtype.Float16},
				})
				convey.So(ok, convey.ShouldBeTrue)

				left, right, out := newFloat16MatMulCase(t, n)
				err := kernel.Run(left, right, out)
				convey.So(err, convey.ShouldBeNil)

				assertFloat16MatMulCase(t, left, right, out, n)
			})
		})
	}
}

func TestMatMulAddFloat16AndBFloat16(t *testing.T) {
	for _, storageDType := range []dtype.DType{dtype.Float16, dtype.BFloat16} {
		storageDType := storageDType

		t.Run(storageDType.Name(), func(t *testing.T) {
			convey.Convey("Output should equal the dtype reference matmul_add", t, func() {
				kernel, ok := Default.Lookup("matmul_add", Signature{
					Layout: tensor.LayoutDense,
					Inputs: []dtype.DType{
						storageDType,
						storageDType,
						storageDType,
					},
					Outputs: []dtype.DType{storageDType},
				})
				convey.So(ok, convey.ShouldBeTrue)

				left, right, bias, out := newDTypeMatMulAddCase(t, storageDType)
				err := kernel.Run(left, right, bias, out)
				convey.So(err, convey.ShouldBeNil)

				assertDTypeMatMulAddCase(t, left, right, bias, out, storageDType)
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

func newDTypeMatMulAddCase(
	testingObject testing.TB,
	storageDType dtype.DType,
) (tensor.Tensor, tensor.Tensor, tensor.Tensor, tensor.Tensor) {
	testingObject.Helper()

	leftShape, _ := tensor.NewShape([]int{3, 5})
	rightShape, _ := tensor.NewShape([]int{5, 4})
	biasShape, _ := tensor.NewShape([]int{4})
	outShape, _ := tensor.NewShape([]int{3, 4})
	left, _ := tensor.NewZeroed(leftShape, storageDType)
	right, _ := tensor.NewZeroed(rightShape, storageDType)
	bias, _ := tensor.NewZeroed(biasShape, storageDType)
	out, _ := tensor.NewZeroed(outShape, storageDType)
	fillDTypeMatMulAddInputs(left, right, bias, storageDType)
	return left, right, bias, out
}

func fillDTypeMatMulAddInputs(
	left tensor.Tensor,
	right tensor.Tensor,
	bias tensor.Tensor,
	storageDType dtype.DType,
) {
	if storageDType == dtype.Float16 {
		fillFloat16MatMulAddInputs(left, right, bias)
		return
	}

	fillBFloat16MatMulAddInputs(left, right, bias)
}

func fillFloat16MatMulAddInputs(left tensor.Tensor, right tensor.Tensor, bias tensor.Tensor) {
	leftView, _ := left.Float16Native()
	rightView, _ := right.Float16Native()
	biasView, _ := bias.Float16Native()

	for index := range leftView {
		leftView[index] = dtype.Fromfloat32(float32(index%9-4) / 8)
	}

	for index := range rightView {
		rightView[index] = dtype.Fromfloat32(float32(index%7-3) / 16)
	}

	for index := range biasView {
		biasView[index] = dtype.Fromfloat32(float32(index%5-2) / 4)
	}
}

func fillBFloat16MatMulAddInputs(left tensor.Tensor, right tensor.Tensor, bias tensor.Tensor) {
	leftView, _ := left.BFloat16Native()
	rightView, _ := right.BFloat16Native()
	biasView, _ := bias.BFloat16Native()

	for index := range leftView {
		leftView[index] = dtype.NewBfloat16FromFloat32(float32(index%9-4) / 8)
	}

	for index := range rightView {
		rightView[index] = dtype.NewBfloat16FromFloat32(float32(index%7-3) / 16)
	}

	for index := range biasView {
		biasView[index] = dtype.NewBfloat16FromFloat32(float32(index%5-2) / 4)
	}
}

func assertDTypeMatMulAddCase(
	testingObject testing.TB,
	left tensor.Tensor,
	right tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
	storageDType dtype.DType,
) {
	testingObject.Helper()

	if storageDType == dtype.Float16 {
		assertFloat16MatMulAddCase(left, right, bias, out)
		return
	}

	assertBFloat16MatMulAddCase(left, right, bias, out)
}

func assertFloat16MatMulAddCase(
	left tensor.Tensor,
	right tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) {
	leftView, _ := left.Float16Native()
	rightView, _ := right.Float16Native()
	biasView, _ := bias.Float16Native()
	outView, _ := out.Float16Native()

	for rowIndex := 0; rowIndex < 3; rowIndex++ {
		for colIndex := 0; colIndex < 4; colIndex++ {
			sum := biasView[colIndex].Float32()

			for innerIndex := 0; innerIndex < 5; innerIndex++ {
				sum += leftView[rowIndex*5+innerIndex].Float32() *
					rightView[innerIndex*4+colIndex].Float32()
			}

			convey.So(outView[rowIndex*4+colIndex], convey.ShouldEqual, dtype.Fromfloat32(sum))
		}
	}
}

func assertBFloat16MatMulAddCase(
	left tensor.Tensor,
	right tensor.Tensor,
	bias tensor.Tensor,
	out tensor.Tensor,
) {
	leftView, _ := left.BFloat16Native()
	rightView, _ := right.BFloat16Native()
	biasView, _ := bias.BFloat16Native()
	outView, _ := out.BFloat16Native()

	for rowIndex := 0; rowIndex < 3; rowIndex++ {
		for colIndex := 0; colIndex < 4; colIndex++ {
			sum := (&biasView[colIndex]).Float32()

			for innerIndex := 0; innerIndex < 5; innerIndex++ {
				sum += (&leftView[rowIndex*5+innerIndex]).Float32() *
					(&rightView[innerIndex*4+colIndex]).Float32()
			}

			convey.So(outView[rowIndex*4+colIndex], convey.ShouldEqual, dtype.NewBfloat16FromFloat32(sum))
		}
	}
}
