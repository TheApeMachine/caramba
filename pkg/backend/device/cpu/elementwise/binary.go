package elementwise

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Elementwise binary kernels: two same-shape tensors in → one out.
The dispatcher routes (a, b, out). Broadcasting lives one layer up;
these kernels require matching shapes.
*/

func runDivFloat32(args ...tensor.Tensor) error {
	left, right, out, err := binaryFloat32Args(args)

	if err != nil {
		return err
	}

	DivFloat32Native(out, left, right)

	return nil
}

func runDivBFloat16(args ...tensor.Tensor) error {
	left, right, out, err := binaryBFloat16Args(args)

	if err != nil {
		return err
	}

	DivBFloat16Native(out, left, right)

	return nil
}

func runMaxBFloat16(args ...tensor.Tensor) error {
	left, right, out, err := binaryBFloat16Args(args)

	if err != nil {
		return err
	}

	MaxBFloat16Native(out, left, right)

	return nil
}

func runMinBFloat16(args ...tensor.Tensor) error {
	left, right, out, err := binaryBFloat16Args(args)

	if err != nil {
		return err
	}

	MinBFloat16Native(out, left, right)

	return nil
}

func runDivFloat16(args ...tensor.Tensor) error {
	left, right, out, err := binaryFloat16Args(args)

	if err != nil {
		return err
	}

	DivFloat16Native(out, left, right)

	return nil
}

func runMaxFloat16(args ...tensor.Tensor) error {
	left, right, out, err := binaryFloat16Args(args)

	if err != nil {
		return err
	}

	MaxFloat16Native(out, left, right)

	return nil
}

func runMinFloat16(args ...tensor.Tensor) error {
	left, right, out, err := binaryFloat16Args(args)

	if err != nil {
		return err
	}

	MinFloat16Native(out, left, right)

	return nil
}

func runMaxFloat32(args ...tensor.Tensor) error {
	left, right, out, err := binaryFloat32Args(args)

	if err != nil {
		return err
	}

	MaxFloat32Native(out, left, right)

	return nil
}

func runMinFloat32(args ...tensor.Tensor) error {
	left, right, out, err := binaryFloat32Args(args)

	if err != nil {
		return err
	}

	MinFloat32Native(out, left, right)

	return nil
}

func binaryFloat32Args(args []tensor.Tensor) (left, right, out []float32, err error) {
	if len(args) != 3 {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	left, err = args[0].Float32Native()

	if err != nil {
		return nil, nil, nil, err
	}

	right, err = args[1].Float32Native()

	if err != nil {
		return nil, nil, nil, err
	}

	out, err = args[2].Float32Native()

	if err != nil {
		return nil, nil, nil, err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	return left, right, out, nil
}

func binaryBFloat16Args(args []tensor.Tensor) (left, right, out []dtype.BF16, err error) {
	if len(args) != 3 {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	left, err = args[0].BFloat16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	right, err = args[1].BFloat16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	out, err = args[2].BFloat16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	return left, right, out, nil
}

func binaryFloat16Args(args []tensor.Tensor) (left, right, out []dtype.F16, err error) {
	if len(args) != 3 {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	left, err = args[0].Float16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	right, err = args[1].Float16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	out, err = args[2].Float16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	return left, right, out, nil
}
