package matmul

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func RunMatMulFloat32(args ...tensor.Tensor) error {
	return runMatMulFloat32(args...)
}

func RunMatMulFloat64(args ...tensor.Tensor) error {
	return runMatMulFloat64(args...)
}

func RunMatMulFloat16(args ...tensor.Tensor) error {
	return runMatMulFloat16(args...)
}

func RunMatMulBFloat16(args ...tensor.Tensor) error {
	return runMatMulBFloat16(args...)
}

func runMatMulFloat32(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	lhs, rhs, out := args[0], args[1], args[2]

	rows, inner, cols, err := matmulDims(lhs, rhs, out)

	if err != nil {
		return err
	}

	leftView, err := lhs.Float32Native()

	if err != nil {
		return err
	}

	rightView, err := rhs.Float32Native()

	if err != nil {
		return err
	}

	outView, err := out.Float32Native()

	if err != nil {
		return err
	}

	for index := range outView {
		outView[index] = 0
	}

	Matmul(
		unsafe.Pointer(&outView[0]),
		unsafe.Pointer(&leftView[0]),
		unsafe.Pointer(&rightView[0]),
		rows, inner, cols,
		dtype.Float32,
	)

	return nil
}

func runMatMulFloat64(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	lhs, rhs, out := args[0], args[1], args[2]

	rows, inner, cols, err := matmulDims(lhs, rhs, out)

	if err != nil {
		return err
	}

	leftView, err := lhs.Float64Native()

	if err != nil {
		return err
	}

	rightView, err := rhs.Float64Native()

	if err != nil {
		return err
	}

	outView, err := out.Float64Native()

	if err != nil {
		return err
	}

	for index := range outView {
		outView[index] = 0
	}

	Matmul(
		unsafe.Pointer(&outView[0]),
		unsafe.Pointer(&leftView[0]),
		unsafe.Pointer(&rightView[0]),
		rows, inner, cols,
		dtype.Float64,
	)

	return nil
}

func runMatMulFloat16(args ...tensor.Tensor) error {
	return runMatMulReducedPrecision(args, dtype.Float16)
}

func runMatMulBFloat16(args ...tensor.Tensor) error {
	return runMatMulReducedPrecision(args, dtype.BFloat16)
}

func runMatMulReducedPrecision(args []tensor.Tensor, format dtype.DType) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	lhs, rhs, out := args[0], args[1], args[2]

	rows, inner, cols, err := matmulDims(lhs, rhs, out)

	if err != nil {
		return err
	}

	leftPointer, err := reducedPrecisionPointer(lhs, format)

	if err != nil {
		return err
	}

	rightPointer, err := reducedPrecisionPointer(rhs, format)

	if err != nil {
		return err
	}

	outPointer, err := reducedPrecisionPointer(out, format)

	if err != nil {
		return err
	}

	Matmul(outPointer, leftPointer, rightPointer, rows, inner, cols, format)

	return nil
}

func reducedPrecisionPointer(value tensor.Tensor, format dtype.DType) (unsafe.Pointer, error) {
	switch format {
	case dtype.Float16:
		view, err := value.Float16Native()

		if err != nil {
			return nil, err
		}

		if len(view) == 0 {
			return unsafe.Pointer(nil), nil
		}

		return unsafe.Pointer(&view[0]), nil
	case dtype.BFloat16:
		view, err := value.BFloat16Native()

		if err != nil {
			return nil, err
		}

		if len(view) == 0 {
			return unsafe.Pointer(nil), nil
		}

		return unsafe.Pointer(&view[0]), nil
	default:
		return nil, tensor.ErrShapeMismatch
	}
}

func matmulDims(lhs, rhs, out tensor.Tensor) (rows, inner, cols int, err error) {
	leftDims := lhs.Shape().Dims()
	rightDims := rhs.Shape().Dims()
	outDims := out.Shape().Dims()

	if len(leftDims) != 2 || len(rightDims) != 2 || len(outDims) != 2 {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	if leftDims[1] != rightDims[0] {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	if outDims[0] != leftDims[0] || outDims[1] != rightDims[1] {
		return 0, 0, 0, tensor.ErrShapeMismatch
	}

	return leftDims[0], leftDims[1], rightDims[1], nil
}
