package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
RoPE (Rotary Position Embedding) — applies a rotation in each pair
of consecutive head dimensions, with frequencies that scale with
position. The standard transformer formulation:

    x'_{2i}   = x_{2i}   * cos(θ_i,p) - x_{2i+1} * sin(θ_i,p)
    x'_{2i+1} = x_{2i}   * sin(θ_i,p) + x_{2i+1} * cos(θ_i,p)

    θ_i,p = p * baseFreq ** (-2i / headDim)

This is the host scalar reference. The NEON / AVX-512 paths land in
follow-up sessions; the kernel signature here is the contract.

Args order: (input, output). Shape: [seq, heads, headDim]. Scalar
parameters (baseFreq, startPosition) are bound at op-plan time via
RoPEConfig and the helper `RunRoPEFloat32` rather than passed as
tensors — orchestrator Phase 11 work will lift them onto the kernel
signature properly when needed.
*/

type RoPEConfig struct {
	BaseFreq      float64
	StartPosition int
}

func DefaultRoPEConfig() RoPEConfig {
	return RoPEConfig{BaseFreq: 10000.0, StartPosition: 0}
}

func init() {
	Default.Register(Kernel{
		Name: "rope",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runRoPEFloat32Default,
	})
}

func runRoPEFloat32Default(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return RunRoPEFloat32(DefaultRoPEConfig(), args[0], args[1])
}

/*
RunRoPEFloat32 applies RoPE with the supplied configuration. Input
must have rank 3 with shape [seqLen, numHeads, headDim] and headDim
must be even.
*/
func RunRoPEFloat32(config RoPEConfig, input, output tensor.Tensor) error {
	dims := input.Shape().Dims()

	if len(dims) != 3 {
		return tensor.ErrShapeMismatch
	}

	seqLen := dims[0]
	numHeads := dims[1]
	headDim := dims[2]

	if headDim%2 != 0 {
		return tensor.ErrShapeMismatch
	}

	if !input.Shape().Equal(output.Shape()) {
		return tensor.ErrShapeMismatch
	}

	inputView, err := input.Float32Native()

	if err != nil {
		return err
	}

	outputView, err := output.Float32Native()

	if err != nil {
		return err
	}

	halfDim := headDim / 2

	for seqIndex := 0; seqIndex < seqLen; seqIndex++ {
		position := float64(seqIndex + config.StartPosition)

		for headIndex := 0; headIndex < numHeads; headIndex++ {
			rowOffset := (seqIndex*numHeads + headIndex) * headDim

			for pairIndex := 0; pairIndex < halfDim; pairIndex++ {
				exponent := -float64(2*pairIndex) / float64(headDim)
				theta := position * math.Pow(config.BaseFreq, exponent)
				cos := float32(math.Cos(theta))
				sin := float32(math.Sin(theta))

				even := inputView[rowOffset+2*pairIndex]
				odd := inputView[rowOffset+2*pairIndex+1]

				outputView[rowOffset+2*pairIndex] = even*cos - odd*sin
				outputView[rowOffset+2*pairIndex+1] = even*sin + odd*cos
			}
		}
	}

	return nil
}
