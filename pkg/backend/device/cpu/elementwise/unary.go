package elementwise

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Elementwise unary kernels: every standard scalar transform with a
[N] in → [N] out signature. SIMD bodies replace the scalar driver
through the per-arch *Native dispatchers in select_*.go.
*/

func unaryBFloat16Args(args []tensor.Tensor) (in, out []dtype.BF16, err error) {
	if len(args) != 2 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	in, err = args[0].BFloat16Native()

	if err != nil {
		return nil, nil, err
	}

	out, err = args[1].BFloat16Native()

	if err != nil {
		return nil, nil, err
	}

	if len(in) != len(out) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return in, out, nil
}

func runAbsBFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryBFloat16Args(args)

	if err != nil {
		return err
	}

	AbsBFloat16Native(out, in)
	return nil
}

func runNegBFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryBFloat16Args(args)

	if err != nil {
		return err
	}

	NegBFloat16Native(out, in)
	return nil
}

func runSqrtBFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryBFloat16Args(args)

	if err != nil {
		return err
	}

	SqrtBFloat16Native(out, in)
	return nil
}

func runReluBFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryBFloat16Args(args)

	if err != nil {
		return err
	}

	ReluBFloat16Native(out, in)
	return nil
}

func unaryFloat16Args(args []tensor.Tensor) (in, out []dtype.F16, err error) {
	if len(args) != 2 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	in, err = args[0].Float16Native()

	if err != nil {
		return nil, nil, err
	}

	out, err = args[1].Float16Native()

	if err != nil {
		return nil, nil, err
	}

	if len(in) != len(out) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return in, out, nil
}

func runAbsFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryFloat16Args(args)

	if err != nil {
		return err
	}

	AbsFloat16Native(out, in)
	return nil
}

func runNegFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryFloat16Args(args)

	if err != nil {
		return err
	}

	NegFloat16Native(out, in)
	return nil
}

func runSqrtFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryFloat16Args(args)

	if err != nil {
		return err
	}

	SqrtFloat16Native(out, in)
	return nil
}

func runReluFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryFloat16Args(args)

	if err != nil {
		return err
	}

	ReluFloat16Native(out, in)
	return nil
}

func unaryFloat32Args(args []tensor.Tensor) (in, out []float32, err error) {
	if len(args) != 2 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	in, err = args[0].Float32Native()

	if err != nil {
		return nil, nil, err
	}

	out, err = args[1].Float32Native()

	if err != nil {
		return nil, nil, err
	}

	if len(in) != len(out) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return in, out, nil
}

func runAbsFloat32(args ...tensor.Tensor) error {
	in, out, err := unaryFloat32Args(args)

	if err != nil {
		return err
	}

	AbsFloat32Native(out, in)

	return nil
}

func runNegFloat32(args ...tensor.Tensor) error {
	in, out, err := unaryFloat32Args(args)

	if err != nil {
		return err
	}

	NegFloat32Native(out, in)

	return nil
}

func runSqrtFloat32(args ...tensor.Tensor) error {
	in, out, err := unaryFloat32Args(args)

	if err != nil {
		return err
	}

	SqrtFloat32Native(out, in)

	return nil
}

func runReluFloat32(args ...tensor.Tensor) error {
	in, out, err := unaryFloat32Args(args)

	if err != nil {
		return err
	}

	ReluFloat32Native(out, in)

	return nil
}
