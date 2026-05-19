package masking

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func runApplyMaskBFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].BFloat16Native()

	if err != nil {
		return err
	}

	mask, err := args[1].BFloat16Native()

	if err != nil {
		return err
	}

	out, err := args[2].BFloat16Native()

	if err != nil {
		return err
	}

	if len(input) != len(mask) || len(out) != len(input) {
		return tensor.ErrShapeMismatch
	}

	if len(input) == 0 {
		return nil
	}

	ApplyMask(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&mask[0]),
		unsafe.Pointer(&out[0]),
		len(input),
		dtype.BFloat16,
	)

	return nil
}

func runApplyMaskFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].Float16Native()

	if err != nil {
		return err
	}

	mask, err := args[1].Float16Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float16Native()

	if err != nil {
		return err
	}

	if len(input) != len(mask) || len(out) != len(input) {
		return tensor.ErrShapeMismatch
	}

	if len(input) == 0 {
		return nil
	}

	ApplyMask(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&mask[0]),
		unsafe.Pointer(&out[0]),
		len(input),
		dtype.Float16,
	)

	return nil
}

func runCausalMaskBFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	out, err := args[1].BFloat16Native()

	if err != nil {
		return err
	}

	dims := args[1].Shape().Dims()

	if len(dims) != 2 {
		return tensor.ErrShapeMismatch
	}

	if dims[0] == 0 || dims[1] == 0 || len(out) == 0 {
		return nil
	}

	CausalMask(unsafe.Pointer(&out[0]), dims[0], dims[1], dtype.BFloat16)

	return nil
}

func runCausalMaskFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	out, err := args[1].Float16Native()

	if err != nil {
		return err
	}

	dims := args[1].Shape().Dims()

	if len(dims) != 2 {
		return tensor.ErrShapeMismatch
	}

	if dims[0] == 0 || dims[1] == 0 || len(out) == 0 {
		return nil
	}

	CausalMask(unsafe.Pointer(&out[0]), dims[0], dims[1], dtype.Float16)

	return nil
}

func runALiBiBiasBFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	scores, err := args[0].BFloat16Native()

	if err != nil {
		return err
	}

	slope, err := args[1].BFloat16Native()

	if err != nil {
		return err
	}

	out, err := args[2].BFloat16Native()

	if err != nil {
		return err
	}

	if len(slope) < 1 || len(scores) < 1 || len(out) != len(scores) {
		return tensor.ErrShapeMismatch
	}

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] == 0 || dims[1] == 0 {
		return tensor.ErrShapeMismatch
	}

	ALiBiBias(
		unsafe.Pointer(&scores[0]),
		unsafe.Pointer(&slope[0]),
		unsafe.Pointer(&out[0]),
		dims[0], dims[1],
		dtype.BFloat16,
	)

	return nil
}

func runALiBiBiasFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	scores, err := args[0].Float16Native()

	if err != nil {
		return err
	}

	slope, err := args[1].Float16Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float16Native()

	if err != nil {
		return err
	}

	if len(slope) < 1 || len(scores) < 1 || len(out) != len(scores) {
		return tensor.ErrShapeMismatch
	}

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] == 0 || dims[1] == 0 {
		return tensor.ErrShapeMismatch
	}

	ALiBiBias(
		unsafe.Pointer(&scores[0]),
		unsafe.Pointer(&slope[0]),
		unsafe.Pointer(&out[0]),
		dims[0], dims[1],
		dtype.Float16,
	)

	return nil
}

func runApplyMask(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	mask, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	if len(input) != len(mask) || len(out) != len(input) {
		return tensor.ErrShapeMismatch
	}

	if len(input) == 0 {
		return nil
	}

	ApplyMask(
		unsafe.Pointer(&input[0]),
		unsafe.Pointer(&mask[0]),
		unsafe.Pointer(&out[0]),
		len(input),
		dtype.Float32,
	)

	return nil
}

func runCausalMask(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	out, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	dims := args[1].Shape().Dims()

	if len(dims) != 2 {
		return tensor.ErrShapeMismatch
	}

	if dims[0] == 0 || dims[1] == 0 || len(out) == 0 {
		return nil
	}

	CausalMask(unsafe.Pointer(&out[0]), dims[0], dims[1], dtype.Float32)

	return nil
}

func runALiBiBias(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	scores, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	slope, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	if len(slope) < 1 || len(scores) < 1 || len(out) != len(scores) {
		return tensor.ErrShapeMismatch
	}

	dims := args[0].Shape().Dims()

	if len(dims) != 2 || dims[0] == 0 || dims[1] == 0 {
		return tensor.ErrShapeMismatch
	}

	ALiBiBias(
		unsafe.Pointer(&scores[0]),
		unsafe.Pointer(&slope[0]),
		unsafe.Pointer(&out[0]),
		dims[0], dims[1],
		dtype.Float32,
	)

	return nil
}
