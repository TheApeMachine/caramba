package kernels

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Add is the example kernel showing the dispatch pattern. Phase 8
populates the full kernel surface (matmul, attention variants,
softmax, layer-norm, rms-norm, gelu, etc.) following the same shape.

The scalar Go body is the reference. SIMD bodies for AVX-512 / AVX2 /
SSE2 / NEON go in add_amd64.s / add_arm64.s in later sessions and
register additional Kernel entries with the appropriate Locations.

Per AGENTS.md §1.2, the SIMD bodies cannot alias each other; each ISA
gets its own kernel body with its own vector instructions.
*/

func init() {
	registerAddFloat32()
	registerAddFloat64()
	registerAddBFloat16()
}

func registerAddFloat32() {
	Default.Register(Kernel{
		Name: "add",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAddFloat32,
	})
}

func registerAddFloat64() {
	Default.Register(Kernel{
		Name: "add",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float64, dtype.Float64},
			Outputs: []dtype.DType{dtype.Float64},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAddFloat64,
	})
}

func registerAddBFloat16() {
	Default.Register(Kernel{
		Name: "add",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runAddBFloat16,
	})
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

	for index := range out {
		out[index] = left[index] + right[index]
	}

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

	for index := range out {
		out[index] = left[index] + right[index]
	}

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

	// bf16 + bf16 with fp32 accumulation per the mixed-dtype
	// kernel convention (§5.5 of TENSOR_BACKEND_REWRITE.md).
	for index := range out {
		sum := (&left[index]).Float32() + (&right[index]).Float32()
		out[index] = dtype.NewBfloat16FromFloat32(sum)
	}

	return nil
}
