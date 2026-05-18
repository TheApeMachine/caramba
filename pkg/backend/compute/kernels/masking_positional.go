package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Masking and positional bias kernels.

  - apply_mask: adds a mask tensor (typically -Inf where masked) to a
    scores tensor.
  - causal_mask: builds a lower-triangular causal mask.
  - alibi: adds ALiBi bias to scores. ALiBi penalizes attention by
    distance using a per-head slope.
*/

func init() {
	Default.Register(Kernel{
		Name: "apply_mask",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runApplyMask,
	})

	Default.Register(Kernel{
		Name: "causal_mask",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runCausalMask,
	})

	Default.Register(Kernel{
		Name: "alibi_bias",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runALiBiBias,
	})

	// bf16 / fp16 mirrors. apply_mask + alibi_bias are
	// widen-compute-narrow; causal_mask writes the dtype's -inf bit
	// pattern (0xFF80 for bf16, 0xFC00 for fp16) directly.
	registerMaskingBFloat16()
	registerMaskingFloat16()
}

func registerMaskingBFloat16() {
	Default.Register(Kernel{
		Name: "apply_mask",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runApplyMaskBFloat16,
	})
	Default.Register(Kernel{
		Name: "causal_mask",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runCausalMaskBFloat16,
	})
	Default.Register(Kernel{
		Name: "alibi_bias",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16, dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runALiBiBiasBFloat16,
	})
}

func registerMaskingFloat16() {
	Default.Register(Kernel{
		Name: "apply_mask",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16, dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runApplyMaskFloat16,
	})
	Default.Register(Kernel{
		Name: "causal_mask",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runCausalMaskFloat16,
	})
	Default.Register(Kernel{
		Name: "alibi_bias",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16, dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runALiBiBiasFloat16,
	})
}

func runApplyMaskBFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].BFloat16Native()
	mask, _ := args[1].BFloat16Native()
	out, _ := args[2].BFloat16Native()

	if len(input) != len(mask) || len(out) != len(input) {
		return tensor.ErrShapeMismatch
	}

	for index := range input {
		sum := (&input[index]).Float32() + (&mask[index]).Float32()
		out[index] = dtype.NewBfloat16FromFloat32(sum)
	}

	return nil
}

func runApplyMaskFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].Float16Native()
	mask, _ := args[1].Float16Native()
	out, _ := args[2].Float16Native()

	if len(input) != len(mask) || len(out) != len(input) {
		return tensor.ErrShapeMismatch
	}

	for index := range input {
		out[index] = dtype.Fromfloat32(input[index].Float32() + mask[index].Float32())
	}

	return nil
}

func runCausalMaskBFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	out, _ := args[1].BFloat16Native()
	dims := args[1].Shape().Dims()

	if len(dims) != 2 {
		return tensor.ErrShapeMismatch
	}

	seqQ := dims[0]
	seqK := dims[1]

	const bf16NegInf = dtype.BF16(0xFF80)

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		for colIndex := 0; colIndex < seqK; colIndex++ {
			if colIndex > rowIndex {
				out[rowIndex*seqK+colIndex] = bf16NegInf
				continue
			}
			out[rowIndex*seqK+colIndex] = 0
		}
	}

	return nil
}

func runCausalMaskFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	out, _ := args[1].Float16Native()
	dims := args[1].Shape().Dims()

	if len(dims) != 2 {
		return tensor.ErrShapeMismatch
	}

	seqQ := dims[0]
	seqK := dims[1]

	const fp16NegInf = dtype.F16(0xFC00)

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		for colIndex := 0; colIndex < seqK; colIndex++ {
			if colIndex > rowIndex {
				out[rowIndex*seqK+colIndex] = fp16NegInf
				continue
			}
			out[rowIndex*seqK+colIndex] = 0
		}
	}

	return nil
}

func runALiBiBiasBFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	scores, _ := args[0].BFloat16Native()
	slope, _ := args[1].BFloat16Native()
	out, _ := args[2].BFloat16Native()

	if len(slope) < 1 || len(out) != len(scores) {
		return tensor.ErrShapeMismatch
	}

	dims := args[0].Shape().Dims()
	if len(dims) != 2 {
		return tensor.ErrShapeMismatch
	}

	seqQ := dims[0]
	seqK := dims[1]
	slopeValue := (&slope[0]).Float32()

	for rowIndex := range seqQ {
		for colIndex := range seqK {
			index := rowIndex*seqK + colIndex
			distance := rowIndex - colIndex

			score := (&scores[index]).Float32()

			if distance < 0 {
				out[index] = scores[index]
				continue
			}

			out[index] = dtype.NewBfloat16FromFloat32(score - slopeValue*float32(distance))
		}
	}

	return nil
}

func runALiBiBiasFloat16(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	scores, _ := args[0].Float16Native()
	slope, _ := args[1].Float16Native()
	out, _ := args[2].Float16Native()

	if len(slope) < 1 || len(out) != len(scores) {
		return tensor.ErrShapeMismatch
	}

	dims := args[0].Shape().Dims()
	if len(dims) != 2 {
		return tensor.ErrShapeMismatch
	}

	seqQ := dims[0]
	seqK := dims[1]
	slopeValue := slope[0].Float32()

	for rowIndex := range seqQ {
		for colIndex := range seqK {
			index := rowIndex*seqK + colIndex
			distance := rowIndex - colIndex

			if distance < 0 {
				out[index] = scores[index]
				continue
			}

			out[index] = dtype.Fromfloat32(scores[index].Float32() - slopeValue*float32(distance))
		}
	}

	return nil
}

/*
runApplyMask adds the mask tensor to the input. Both must have
matching shape; typical use is scores + mask where mask carries
-Inf for blocked positions.
*/
func runApplyMask(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, _ := args[0].Float32Native()
	mask, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(input) != len(mask) || len(out) != len(input) {
		return tensor.ErrShapeMismatch
	}

	for index, value := range input {
		out[index] = value + mask[index]
	}

	return nil
}

/*
runCausalMask fills the output with a lower-triangular mask of shape
[seqQ, seqK]: 0 on the lower triangle (including diagonal), -Inf
above. The output's shape determines seqQ and seqK; the input is a
zero placeholder for the kernel-registry signature.
*/
func runCausalMask(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	out, _ := args[1].Float32Native()

	dims := args[1].Shape().Dims()

	if len(dims) != 2 {
		return tensor.ErrShapeMismatch
	}

	seqQ := dims[0]
	seqK := dims[1]

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		for colIndex := 0; colIndex < seqK; colIndex++ {
			if colIndex > rowIndex {
				out[rowIndex*seqK+colIndex] = float32(math.Inf(-1))
				continue
			}

			out[rowIndex*seqK+colIndex] = 0
		}
	}

	return nil
}

/*
runALiBiBias adds an ALiBi bias to a [seqQ, seqK] scores tensor.
Bias[q, k] = -slope × (q - k) for k ≤ q, 0 otherwise. The slope is
read from a scalar tensor (length 1) — typically the per-head slope
selected from the geometric series ALiBi uses.
*/
func runALiBiBias(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	scores, _ := args[0].Float32Native()
	slope, _ := args[1].Float32Native()
	out, _ := args[2].Float32Native()

	if len(slope) < 1 || len(out) != len(scores) {
		return tensor.ErrShapeMismatch
	}

	dims := args[0].Shape().Dims()

	if len(dims) != 2 {
		return tensor.ErrShapeMismatch
	}

	seqQ := dims[0]
	seqK := dims[1]
	slopeValue := slope[0]

	for rowIndex := range seqQ {
		for colIndex := range seqK {
			index := rowIndex*seqK + colIndex
			distance := rowIndex - colIndex

			if distance < 0 {
				out[index] = scores[index]
				continue
			}

			out[index] = scores[index] - slopeValue*float32(distance)
		}
	}

	return nil
}
