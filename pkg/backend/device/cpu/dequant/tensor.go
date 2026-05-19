package dequant

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

func RunInt8DequantDefault(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return Int8Float32(DefaultInt8Config(), args[0], args[1])
}

func RunInt4DequantDefault(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return Int4Float32(DefaultInt4Config(), args[0], args[1])
}

func Int8Float32(
	config Int8Config,
	quantized, output tensor.Tensor,
) error {
	quantView, err := quantized.Int8Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	if len(outView) != len(quantView) {
		return tensor.ErrShapeMismatch
	}

	DequantInt8Native(outView, quantView, config.Scale, config.ZeroPoint)

	return nil
}

func Int4Float32(
	config Int4Config,
	quantized, output tensor.Tensor,
) error {
	pairs, err := quantized.Int4Native()

	if err != nil {
		return err
	}

	outView, err := output.Float32Native()

	if err != nil {
		return err
	}

	if len(outView) > pairs.Len() {
		return tensor.ErrShapeMismatch
	}

	DequantInt4Native(outView, pairs, config.Scale, config.ZeroPoint)

	return nil
}
