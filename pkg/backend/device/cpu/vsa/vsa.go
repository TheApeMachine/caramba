package vsa

import (
	"github.com/theapemachine/manifesto/tensor"
)

/*
Vector Symbolic Architecture (VSA) primitives: bind (Hadamard
product / circular convolution), bundle (superposition / sum),
permute, inverse_permute.

  - bind:    elementwise multiplication of two hypervectors.
  - bundle:  elementwise sum (often normalized) of two hypervectors.
  - permute: cyclic shift of a hypervector by config.Shift positions.
  - inverse_permute: cyclic shift in the opposite direction.

These primitives compose into vector-symbolic memory operations
that the platform's research workloads use directly.
*/

type VSAConfig struct {
	Shift int
}

func DefaultVSAConfig() VSAConfig {
	return VSAConfig{Shift: 1}
}

func runVSABind(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	leftView, _ := args[0].Float32Native()
	rightView, _ := args[1].Float32Native()
	outView, _ := args[2].Float32Native()

	if len(leftView) != len(rightView) || len(outView) != len(leftView) {
		return tensor.ErrShapeMismatch
	}

	VsaBindFloat32Native(outView, leftView, rightView)

	return nil
}

func runVSABundle(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	leftView, _ := args[0].Float32Native()
	rightView, _ := args[1].Float32Native()
	outView, _ := args[2].Float32Native()

	if len(leftView) != len(rightView) || len(outView) != len(leftView) {
		return tensor.ErrShapeMismatch
	}

	VsaBundleFloat32Native(outView, leftView, rightView)

	return nil
}

func runVSAPermuteDefault(args ...tensor.Tensor) error {
	return VSAPermute(DefaultVSAConfig(), args[0], args[1])
}

func runVSAInversePermuteDefault(args ...tensor.Tensor) error {
	config := DefaultVSAConfig()
	config.Shift = -config.Shift
	return VSAPermute(config, args[0], args[1])
}

/*
VSAPermute performs a cyclic shift of length config.Shift. Negative
shifts rotate the opposite direction.
*/
func VSAPermute(config VSAConfig, input, output tensor.Tensor) error {
	inView, _ := input.Float32Native()
	outView, _ := output.Float32Native()

	if len(inView) != len(outView) {
		return tensor.ErrShapeMismatch
	}

	if len(inView) == 0 {
		return nil
	}

	shift := config.Shift % len(inView)

	if shift < 0 {
		shift += len(inView)
	}

	VsaPermuteFloat32Native(outView, inView, shift)

	return nil
}

func runVSASimilarity(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	leftView, _ := args[0].Float32Native()
	rightView, _ := args[1].Float32Native()
	outView, _ := args[2].Float32Native()

	if len(leftView) != len(rightView) || len(outView) < 1 {
		return tensor.ErrShapeMismatch
	}

	outView[0] = VsaSimilarityFloat32Native(leftView, rightView)

	return nil
}
