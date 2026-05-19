package rope

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func RunRoPEFloat32Default(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return RunRoPEFloat32(DefaultRoPEConfig(), args[0], args[1])
}

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

	RoPE(
		config,
		unsafe.Pointer(&inputView[0]),
		unsafe.Pointer(&outputView[0]),
		seqLen,
		numHeads,
		headDim,
		dtype.Float32,
	)

	return nil
}
