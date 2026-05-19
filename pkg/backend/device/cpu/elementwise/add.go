package elementwise

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Add implements dense element-wise addition for float64, float32,
float16, and bfloat16. SIMD bodies are selected through the per-arch
*Native dispatchers in select_*.go.
*/

func runAddFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	left, err := args[0].Float16Native()

	if err != nil {
		return err
	}

	right, err := args[1].Float16Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float16Native()

	if err != nil {
		return err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return tensor.ErrShapeMismatch
	}

	Add(
		unsafe.Pointer(&out[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(out),
		dtype.Float16,
	)
	return nil
}

func runAddFloat32(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	left, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	right, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return tensor.ErrShapeMismatch
	}

	Add(
		unsafe.Pointer(&out[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(out),
		dtype.Float32,
	)

	return nil
}

func runAddFloat64(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	left, err := args[0].Float64Native()

	if err != nil {
		return err
	}

	right, err := args[1].Float64Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float64Native()

	if err != nil {
		return err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return tensor.ErrShapeMismatch
	}

	AddFloat64(
		unsafe.Pointer(&out[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(out),
	)

	return nil
}

func runAddBFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	left, err := args[0].BFloat16Native()

	if err != nil {
		return err
	}

	right, err := args[1].BFloat16Native()

	if err != nil {
		return err
	}

	out, err := args[2].BFloat16Native()

	if err != nil {
		return err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return tensor.ErrShapeMismatch
	}

	Add(
		unsafe.Pointer(&out[0]),
		unsafe.Pointer(&left[0]),
		unsafe.Pointer(&right[0]),
		len(out),
		dtype.BFloat16,
	)

	return nil
}
