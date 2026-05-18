package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Elementwise unary kernels: every standard scalar transform with a
[N] in → [N] out signature. Each one registers under its canonical
op name and routes through a generic unaryFloat32 driver that takes
a per-element function. SIMD bodies replace the driver in later
sessions; the dispatch shape is the same.
*/

type unaryOp func(float32) float32

func unaryFloat32(op unaryOp) func(args ...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
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
			out[index] = op(value)
		}

		return nil
	}
}

func registerUnary(name string, op unaryOp) {
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       unaryFloat32(op),
	})
}

func init() {
	registerUnary("abs", func(value float32) float32 { return float32(math.Abs(float64(value))) })
	registerUnary("neg", func(value float32) float32 { return -value })
	registerUnary("square", func(value float32) float32 { return value * value })
	registerUnary("sqrt", func(value float32) float32 { return float32(math.Sqrt(float64(value))) })
	registerUnary("rsqrt", func(value float32) float32 {
		return float32(1.0 / math.Sqrt(float64(value)))
	})
	registerUnary("recip", func(value float32) float32 { return 1 / value })
	registerUnary("exp", func(value float32) float32 { return float32(math.Exp(float64(value))) })
	registerUnary("log", func(value float32) float32 { return float32(math.Log(float64(value))) })
	registerUnary("log1p", func(value float32) float32 { return float32(math.Log1p(float64(value))) })
	registerUnary("expm1", func(value float32) float32 { return float32(math.Expm1(float64(value))) })
	registerUnary("sin", func(value float32) float32 { return float32(math.Sin(float64(value))) })
	registerUnary("cos", func(value float32) float32 { return float32(math.Cos(float64(value))) })
	registerUnary("tan", func(value float32) float32 { return float32(math.Tan(float64(value))) })
	registerUnary("asin", func(value float32) float32 { return float32(math.Asin(float64(value))) })
	registerUnary("acos", func(value float32) float32 { return float32(math.Acos(float64(value))) })
	registerUnary("atan", func(value float32) float32 { return float32(math.Atan(float64(value))) })
	registerUnary("sinh", func(value float32) float32 { return float32(math.Sinh(float64(value))) })
	registerUnary("cosh", func(value float32) float32 { return float32(math.Cosh(float64(value))) })
	registerUnary("tanh", func(value float32) float32 { return float32(math.Tanh(float64(value))) })
	registerUnary("erf", func(value float32) float32 { return float32(math.Erf(float64(value))) })
	registerUnary("erfc", func(value float32) float32 { return float32(math.Erfc(float64(value))) })
	registerUnary("ceil", func(value float32) float32 { return float32(math.Ceil(float64(value))) })
	registerUnary("floor", func(value float32) float32 { return float32(math.Floor(float64(value))) })
	registerUnary("round", func(value float32) float32 { return float32(math.Round(float64(value))) })
	registerUnary("trunc", func(value float32) float32 { return float32(math.Trunc(float64(value))) })
	registerUnary("sign", func(value float32) float32 {
		switch {
		case value > 0:
			return 1
		case value < 0:
			return -1
		}
		return 0
	})
}
