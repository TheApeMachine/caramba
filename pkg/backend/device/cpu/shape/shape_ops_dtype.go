package shape

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Mixed-precision dispatch for shape-manipulation ops. These ops are
essentially dtype-agnostic data movement (gather/scatter/transpose
copy bytes; where/masked_fill conditionally select bytes), but they
register with explicit dtype signatures so the kernel registry's
strict signature match works for bf16/fp16 tensors.

The bf16/fp16 entries route through the existing f32 runners via the
same allocate-temp-f32-tensor trick used by the conv/pool dispatchers.
A future optimization can replace this with direct byte-level copies
that skip the widen/narrow entirely.
*/

// runShapeUnaryMixed handles ops with signature (input, output) where
// both are at the same paramDType.
func runShapeUnaryMixed(
	args []tensor.Tensor,
	kind dtype.DType,
	f32Runner func(args ...tensor.Tensor) error,
) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	inTemp, err := tensor.NewZeroed(args[0].Shape(), dtype.Float32)

	if err != nil {
		return err
	}

	outTemp, err := tensor.NewZeroed(args[1].Shape(), dtype.Float32)

	if err != nil {
		return err
	}

	inView, _ := inTemp.Float32Native()

	if err := widenToF32(args[0], kind, inView); err != nil {
		return err
	}

	if err := f32Runner(inTemp, outTemp); err != nil {
		return err
	}

	outView, _ := outTemp.Float32Native()
	return narrowFromF32(args[1], kind, outView)
}

// runShapeOpWithIntIndex handles ops with signature (data paramDType,
// indices Int32, output paramDType). The paramDtype tensor is at the
// given paramIndex among args.
func runShapeOpWithIntIndex(
	args []tensor.Tensor,
	kind dtype.DType,
	f32Runner func(args ...tensor.Tensor) error,
	paramIndex int,
) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	inTemp, err := tensor.NewZeroed(args[paramIndex].Shape(), dtype.Float32)

	if err != nil {
		return err
	}

	outTemp, err := tensor.NewZeroed(args[2].Shape(), dtype.Float32)

	if err != nil {
		return err
	}

	inView, _ := inTemp.Float32Native()

	if err := widenToF32(args[paramIndex], kind, inView); err != nil {
		return err
	}

	// Build the rewritten argument list with f32 in place of bf16/fp16.
	rewritten := make([]tensor.Tensor, 3)

	copy(rewritten, args)

	rewritten[paramIndex] = inTemp
	rewritten[2] = outTemp

	if err := f32Runner(rewritten...); err != nil {
		return err
	}

	outView, _ := outTemp.Float32Native()
	return narrowFromF32(args[2], kind, outView)
}

// runScatterMixed handles scatter: (data paramDType, indices Int32,
// updates paramDType) → output paramDType.
func runScatterMixed(args []tensor.Tensor, kind dtype.DType) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	dataTemp, err := tensor.NewZeroed(args[0].Shape(), dtype.Float32)

	if err != nil {
		return err
	}

	updatesTemp, err := tensor.NewZeroed(args[2].Shape(), dtype.Float32)

	if err != nil {
		return err
	}

	outTemp, err := tensor.NewZeroed(args[3].Shape(), dtype.Float32)

	if err != nil {
		return err
	}

	dataView, _ := dataTemp.Float32Native()
	updatesView, _ := updatesTemp.Float32Native()

	if err := widenToF32(args[0], kind, dataView); err != nil {
		return err
	}
	if err := widenToF32(args[2], kind, updatesView); err != nil {
		return err
	}

	if err := runScatterFloat32Int32(dataTemp, args[1], updatesTemp, outTemp); err != nil {
		return err
	}

	outView, _ := outTemp.Float32Native()
	return narrowFromF32(args[3], kind, outView)
}

// runSliceMixed handles slice: (input, dim, start, end) → output.
func runSliceMixed(args []tensor.Tensor, kind dtype.DType) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	inTemp, err := tensor.NewZeroed(args[0].Shape(), dtype.Float32)
	if err != nil {
		return err
	}

	outTemp, err := tensor.NewZeroed(args[4].Shape(), dtype.Float32)
	if err != nil {
		return err
	}

	inView, _ := inTemp.Float32Native()

	if err := widenToF32(args[0], kind, inView); err != nil {
		return err
	}

	if err := runSlice(inTemp, args[1], args[2], args[3], outTemp); err != nil {
		return err
	}

	outView, _ := outTemp.Float32Native()
	return narrowFromF32(args[4], kind, outView)
}

// runConcatMixed handles concat: (a, b) → output, all at paramDType.
func runConcatMixed(args []tensor.Tensor, kind dtype.DType) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	aTemp, err := tensor.NewZeroed(args[0].Shape(), dtype.Float32)

	if err != nil {
		return err
	}

	bTemp, err := tensor.NewZeroed(args[1].Shape(), dtype.Float32)

	if err != nil {
		return err
	}

	outTemp, err := tensor.NewZeroed(args[2].Shape(), dtype.Float32)

	if err != nil {
		return err
	}

	aView, _ := aTemp.Float32Native()
	bView, _ := bTemp.Float32Native()

	if err := widenToF32(args[0], kind, aView); err != nil {
		return err
	}
	if err := widenToF32(args[1], kind, bView); err != nil {
		return err
	}

	if err := runConcatFloat32(aTemp, bTemp, outTemp); err != nil {
		return err
	}

	outView, _ := outTemp.Float32Native()
	return narrowFromF32(args[2], kind, outView)
}
