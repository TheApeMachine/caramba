package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Elementwise kernels: mul, sub, gelu, relu. Per AGENTS.md §1 every
kernel ships full implementations across the required execution
targets at equal standing — scalar Go reference, AVX-512 / AVX2 /
SSE2 (amd64), NEON (arm64), Metal, CUDA, and XLA. The registrations
in this file are the scalar Go entry points; the SIMD and device
paths register their own Kernel entries with matching signatures
and dispatch through Default.Lookup.
*/

func init() {
	registerMulFloat32()
	registerMulBFloat16()
	registerMulFloat16()
	registerSubFloat32()
	registerSubBFloat16()
	registerSubFloat16()
	registerGELUFloat32()
	registerReLUFloat32()
	registerReLUBFloat16()
}

func registerMulFloat16() {
	Default.Register(Kernel{
		Name: "mul",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16, dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMulFloat16,
	})
}

func registerSubFloat16() {
	Default.Register(Kernel{
		Name: "sub",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16, dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runSubFloat16,
	})
}

func runMulFloat16(args ...tensor.Tensor) error {
	left, right, out, err := binaryFloat16Args(args)

	if err != nil {
		return err
	}

	mulFloat16Native(out, left, right)
	return nil
}

func runSubFloat16(args ...tensor.Tensor) error {
	left, right, out, err := binaryFloat16Args(args)

	if err != nil {
		return err
	}

	subFloat16Native(out, left, right)
	return nil
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

func registerMulBFloat16() {
	Default.Register(Kernel{
		Name: "mul",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMulBFloat16,
	})
}

func registerSubBFloat16() {
	Default.Register(Kernel{
		Name: "sub",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runSubBFloat16,
	})
}

func runMulBFloat16(args ...tensor.Tensor) error {
	left, right, out, err := binaryBFloat16Args(args)

	if err != nil {
		return err
	}

	mulBFloat16Native(out, left, right)
	return nil
}

func runSubBFloat16(args ...tensor.Tensor) error {
	left, right, out, err := binaryBFloat16Args(args)

	if err != nil {
		return err
	}

	subBFloat16Native(out, left, right)
	return nil
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

func registerMulFloat32() {
	Default.Register(Kernel{
		Name: "mul",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runMulFloat32,
	})
}

func registerSubFloat32() {
	Default.Register(Kernel{
		Name: "sub",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runSubFloat32,
	})
}

func registerGELUFloat32() {
	Default.Register(Kernel{
		Name: "gelu",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runGELUFloat32,
	})
}

func registerReLUFloat32() {
	Default.Register(Kernel{
		Name: "relu",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runReLUFloat32,
	})
}

func registerReLUBFloat16() {
	Default.Register(Kernel{
		Name: "relu",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runReluBFloat16,
	})
}

func runMulFloat32(args ...tensor.Tensor) error {
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

	mulFloat32Native(out, left, right)

	return nil
}

func runSubFloat32(args ...tensor.Tensor) error {
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

	subFloat32Native(out, left, right)

	return nil
}

func runGELUFloat32(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	if len(input) != len(out) {
		return tensor.ErrShapeMismatch
	}

	// Exact GELU per AGENTS.md: 0.5 * x * (1 + erf(x / sqrt(2))).
	// The tanh approximation is forbidden unless explicitly requested.
	const sqrtTwo = 1.41421356237309504880

	for index, value := range input {
		erfArgument := float64(value) / sqrtTwo
		out[index] = 0.5 * value * float32(1+math.Erf(erfArgument))
	}

	return nil
}

func runReLUFloat32(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	if len(input) != len(out) {
		return tensor.ErrShapeMismatch
	}

	reluFloat32Native(out, input)

	return nil
}
