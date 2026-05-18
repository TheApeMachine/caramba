package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Elementwise kernels: mul, sub, gelu, relu. Each registers float32 +
float64 + bf16 paths; SIMD variants land alongside in later sessions.
*/

func init() {
	registerMulFloat32()
	registerSubFloat32()
	registerGELUFloat32()
	registerReLUFloat32()
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

	for index := range out {
		out[index] = left[index] * right[index]
	}

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

	for index := range out {
		out[index] = left[index] - right[index]
	}

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

	for index, value := range input {
		if value > 0 {
			out[index] = value
			continue
		}

		out[index] = 0
	}

	return nil
}
