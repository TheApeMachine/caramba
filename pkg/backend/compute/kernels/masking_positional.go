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
