package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
MatMul registers the dense matrix-multiply kernel for the host
backend. Five-host-ISA SIMD bodies land alongside this scalar Go
reference in later sessions; the registrations here advertise the
scalar paths first so the dispatch table is populated end-to-end.

The convention follows the doc's §5.5: mixed-dtype matmul accumulates
in float32 even when inputs are bf16, then narrows on write-back if
the output dtype is bf16.

Inputs to Run: (lhs, rhs, output). lhs is shape [M, K]; rhs is shape
[K, N]; output is shape [M, N]. All three must agree on dtype for
the same-dtype kernels (Float64-Float64-Float64, Float32-Float32-
Float32) and on the mixed-precision dtype tuple for the bf16 path
(BFloat16-BFloat16-BFloat16 with fp32 accumulation internally).
*/

func init() {
	registerMatMulFloat32()
	registerMatMulFloat16()
	registerMatMulFloat64()
	registerMatMulBFloat16()
}

func registerMatMulFloat32() {
	Default.Register(Kernel{
		Name: "matmul",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMatMulFloat32,
	})
}

func registerMatMulFloat64() {
	Default.Register(Kernel{
		Name: "matmul",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float64, dtype.Float64},
			Outputs: []dtype.DType{dtype.Float64},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMatMulFloat64,
	})
}

func registerMatMulFloat16() {
	Default.Register(Kernel{
		Name: "matmul",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16, dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMatMulFloat16,
	})
}

func registerMatMulBFloat16() {
	Default.Register(Kernel{
		Name: "matmul",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMatMulBFloat16,
	})
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

	matmulFloat32Native(outView, leftView, rightView, rows, inner, cols)
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

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		for innerIndex := 0; innerIndex < inner; innerIndex++ {
			leftValue := leftView[rowIndex*inner+innerIndex]

			for colIndex := 0; colIndex < cols; colIndex++ {
				outView[rowIndex*cols+colIndex] +=
					leftValue * rightView[innerIndex*cols+colIndex]
			}
		}
	}

	return nil
}

func runMatMulFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	lhs, rhs, out := args[0], args[1], args[2]

	rows, inner, cols, err := matmulDims(lhs, rhs, out)
	if err != nil {
		return err
	}

	leftView, err := lhs.Float16Native()
	if err != nil {
		return err
	}

	rightView, err := rhs.Float16Native()
	if err != nil {
		return err
	}

	outView, err := out.Float16Native()
	if err != nil {
		return err
	}

	// Mixed-dtype matmul per §5.5: f32 accumulation. Same widen→axpy→
	// narrow approach as BF16 matmul, but via FCVTL/FCVTN for fp16
	// widening/narrowing.
	leftF32 := borrowFloat32Buffer(rows * inner)
	rightF32 := borrowFloat32Buffer(inner * cols)
	accumulator := borrowFloat32Buffer(rows * cols)

	defer releaseFloat32Buffer(leftF32)
	defer releaseFloat32Buffer(rightF32)
	defer releaseFloat32Buffer(accumulator)

	float16BulkToFloat32(leftF32, leftView)
	float16BulkToFloat32(rightF32, rightView)

	for index := range accumulator {
		accumulator[index] = 0
	}

	for rowIndex := range rows {
		rowOffset := rowIndex * cols
		accRow := accumulator[rowOffset : rowOffset+cols]

		for innerIndex := range inner {
			leftValue := leftF32[rowIndex*inner+innerIndex]
			rightRow := rightF32[innerIndex*cols : innerIndex*cols+cols]

			axpyFloat32Native(accRow, rightRow, leftValue)
		}
	}

	float32BulkToFloat16(outView, accumulator)

	return nil
}

func runMatMulBFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	lhs, rhs, out := args[0], args[1], args[2]

	rows, inner, cols, err := matmulDims(lhs, rhs, out)

	if err != nil {
		return err
	}

	leftView, err := lhs.BFloat16Native()

	if err != nil {
		return err
	}

	rightView, err := rhs.BFloat16Native()

	if err != nil {
		return err
	}

	outView, err := out.BFloat16Native()

	if err != nil {
		return err
	}

	// Mixed-dtype matmul per §5.5: f32 accumulation throughout.
	// Strategy: bulk-widen both operands to f32 via NEON, run a
	// rank-1 update kernel against an f32 accumulator (each k step
	// uses an axpy-style inner loop that the compiler vectorizes via
	// addFloat32Native), narrow the accumulator back to bf16.
	leftF32 := borrowFloat32Buffer(rows * inner)
	rightF32 := borrowFloat32Buffer(inner * cols)
	accumulator := borrowFloat32Buffer(rows * cols)

	defer releaseFloat32Buffer(leftF32)
	defer releaseFloat32Buffer(rightF32)
	defer releaseFloat32Buffer(accumulator)

	bfloat16BulkToFloat32(leftF32, leftView)
	bfloat16BulkToFloat32(rightF32, rightView)

	for index := range accumulator {
		accumulator[index] = 0
	}

	for rowIndex := range rows {
		rowOffset := rowIndex * cols
		accRow := accumulator[rowOffset : rowOffset+cols]

		for innerIndex := range inner {
			leftValue := leftF32[rowIndex*inner+innerIndex]
			rightRow := rightF32[innerIndex*cols : innerIndex*cols+cols]

			// accRow += leftValue * rightRow — single-pass NEON axpy.
			axpyFloat32Native(accRow, rightRow, leftValue)
		}
	}

	float32BulkToBFloat16(outView, accumulator)

	return nil
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
