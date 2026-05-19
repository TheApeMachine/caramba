package quant

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/dequant"
)

func RunInt8QuantDefault(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return Int8Float32(dequant.DefaultInt8Config(), args[0], args[1])
}

func Int8Float32(
	config dequant.Int8Config,
	input, output tensor.Tensor,
) error {
	inputView, err := input.Float32Native()

	if err != nil {
		return err
	}

	outView, err := output.Int8Native()

	if err != nil {
		return err
	}

	if len(outView) != len(inputView) {
		return tensor.ErrShapeMismatch
	}

	QuantInt8Native(outView, inputView, config.Scale, config.ZeroPoint)

	return nil
}
