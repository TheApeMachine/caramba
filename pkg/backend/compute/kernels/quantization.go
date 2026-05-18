package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Quantization / dequantization kernels for int8 and int4 inference
paths (Phase 8.5). Standard scale-and-zero-point convention:

    quantized = round(real / scale) + zeroPoint
    real      = (quantized - zeroPoint) * scale

For int4, two values pack per byte via dtype.Int4Pair; per-group
scales are common in GPTQ/AWQ formats and the kernels here accept
per-tensor scale + zero point. Per-group support lands in a follow-up
when the loaders for those formats are implemented.

Args order for int8_dequant: (quantized [Int8], output [Float32]).
The scale and zeroPoint are bound at call time via the typed
DequantInt8Float32 helper.
*/

type DequantInt8Config struct {
	Scale     float32
	ZeroPoint int8
}

type DequantInt4Config struct {
	Scale     float32
	ZeroPoint int8
}

func DefaultDequantInt8Config() DequantInt8Config {
	return DequantInt8Config{Scale: 1.0, ZeroPoint: 0}
}

func DefaultDequantInt4Config() DequantInt4Config {
	return DequantInt4Config{Scale: 1.0, ZeroPoint: 0}
}

func init() {
	Default.Register(Kernel{
		Name: "int8_dequant",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Int8},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runInt8DequantDefault,
	})

	Default.Register(Kernel{
		Name: "int4_dequant",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Int4},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runInt4DequantDefault,
	})

	Default.Register(Kernel{
		Name: "int8_quant",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Int8},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runInt8QuantDefault,
	})
}

/*
DequantInt8Float32 applies scale + zero-point dequantization on int8
storage.
*/
func DequantInt8Float32(
	config DequantInt8Config,
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

	for index, value := range quantView {
		outView[index] = float32(int(value)-int(config.ZeroPoint)) * config.Scale
	}

	return nil
}

func runInt8DequantDefault(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return DequantInt8Float32(DefaultDequantInt8Config(), args[0], args[1])
}

/*
DequantInt4Float32 applies scale + zero-point dequantization on
packed int4 storage. Output length must equal 2 × input byte length
or 2 × input byte length − 1 when the logical count is odd.
*/
func DequantInt4Float32(
	config DequantInt4Config,
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

	for index := range outView {
		nibble := pairs.Get(index)
		outView[index] = float32(int(nibble)-int(config.ZeroPoint)) * config.Scale
	}

	return nil
}

func runInt4DequantDefault(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return DequantInt4Float32(DefaultDequantInt4Config(), args[0], args[1])
}

/*
QuantInt8Float32 quantizes float32 values to int8 with the supplied
scale + zero-point. Outputs saturate at math.MinInt8 / math.MaxInt8.
*/
func QuantInt8Float32(
	config DequantInt8Config,
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

	for index, value := range inputView {
		scaled := math.Round(float64(value/config.Scale)) + float64(config.ZeroPoint)

		switch {
		case scaled > float64(math.MaxInt8):
			outView[index] = math.MaxInt8
		case scaled < float64(math.MinInt8):
			outView[index] = math.MinInt8
		default:
			outView[index] = int8(scaled)
		}
	}

	return nil
}

func runInt8QuantDefault(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return QuantInt8Float32(DefaultDequantInt8Config(), args[0], args[1])
}
