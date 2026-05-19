package neon

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

// neonSliceUnaryOverrides lists op names whose f32 path is registered
// elsewhere as a NEON slice runner. registerUnary skips the f32
// registration for these names so the NEON entry isn't duplicated;
// the bf16/fp16 auto-registrations still fire and widen-route-narrow
// through the NEON f32 backend.
var neonSliceUnaryOverrides = map[string]bool{
	"exp":       true,
	"sigmoid":   true,
	"silu":      true,
	"swish":     true,
	"tanh":      true,
	"log":       true,
	"gelu_tanh": true,
}

func registerUnary(name string, op unaryOp) {
	if neonSliceUnaryOverrides[name] {
		// Skip all three dtype registrations; the dedicated NEON
		// slice-runner file (activations_neon_register.go) handles
		// f32 + bf16 + fp16 for these names.
		return
	}

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
	// Mirror every f32 unary as bf16 and fp16 by widen-compute-narrow.
	// Mathematical contract per §5.5: f32 evaluation of the op, bf16/fp16
	// rounding only on the final write-back.
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       unaryBFloat16(op),
	})
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       unaryFloat16Generic(op),
	})
}

func unaryBFloat16(op unaryOp) func(args ...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		input, out, err := unaryBFloat16Args(args)

		if err != nil {
			return err
		}

		for index := range input {
			value := (&input[index]).Float32()
			out[index] = dtype.NewBfloat16FromFloat32(op(value))
		}

		return nil
	}
}

func unaryFloat16Generic(op unaryOp) func(args ...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		input, out, err := unaryFloat16Args(args)

		if err != nil {
			return err
		}

		for index := range input {
			value := input[index].Float32()
			out[index] = dtype.Fromfloat32(op(value))
		}

		return nil
	}
}

func registerUnarySIMD(name string, run func(args ...tensor.Tensor) error) {
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       run,
	})
}

func registerUnarySIMDBFloat16(name string, run func(args ...tensor.Tensor) error) {
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       run,
	})
}

func unaryBFloat16Args(args []tensor.Tensor) (in, out []dtype.BF16, err error) {
	if len(args) != 2 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	in, err = args[0].BFloat16Native()

	if err != nil {
		return nil, nil, err
	}

	out, err = args[1].BFloat16Native()

	if err != nil {
		return nil, nil, err
	}

	if len(in) != len(out) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return in, out, nil
}

func runAbsBFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryBFloat16Args(args)

	if err != nil {
		return err
	}

	AbsBFloat16Native(out, in)
	return nil
}

func runNegBFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryBFloat16Args(args)

	if err != nil {
		return err
	}

	NegBFloat16Native(out, in)
	return nil
}

func runSqrtBFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryBFloat16Args(args)

	if err != nil {
		return err
	}

	SqrtBFloat16Native(out, in)
	return nil
}

func runReluBFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryBFloat16Args(args)

	if err != nil {
		return err
	}

	ReluBFloat16Native(out, in)
	return nil
}

func registerUnarySIMDFloat16(name string, run func(args ...tensor.Tensor) error) {
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       run,
	})
}

func unaryFloat16Args(args []tensor.Tensor) (in, out []dtype.F16, err error) {
	if len(args) != 2 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	in, err = args[0].Float16Native()

	if err != nil {
		return nil, nil, err
	}

	out, err = args[1].Float16Native()

	if err != nil {
		return nil, nil, err
	}

	if len(in) != len(out) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return in, out, nil
}

func runAbsFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryFloat16Args(args)

	if err != nil {
		return err
	}

	AbsFloat16Native(out, in)
	return nil
}

func runNegFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryFloat16Args(args)

	if err != nil {
		return err
	}

	NegFloat16Native(out, in)
	return nil
}

func runSqrtFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryFloat16Args(args)

	if err != nil {
		return err
	}

	SqrtFloat16Native(out, in)
	return nil
}

func runReluFloat16(args ...tensor.Tensor) error {
	in, out, err := unaryFloat16Args(args)

	if err != nil {
		return err
	}

	ReluFloat16Native(out, in)
	return nil
}

func unaryFloat32Args(args []tensor.Tensor) (in, out []float32, err error) {
	if len(args) != 2 {
		return nil, nil, tensor.ErrShapeMismatch
	}

	in, err = args[0].Float32Native()

	if err != nil {
		return nil, nil, err
	}

	out, err = args[1].Float32Native()

	if err != nil {
		return nil, nil, err
	}

	if len(in) != len(out) {
		return nil, nil, tensor.ErrShapeMismatch
	}

	return in, out, nil
}

func runAbsFloat32(args ...tensor.Tensor) error {
	in, out, err := unaryFloat32Args(args)

	if err != nil {
		return err
	}

	AbsFloat32Native(out, in)

	return nil
}

func runNegFloat32(args ...tensor.Tensor) error {
	in, out, err := unaryFloat32Args(args)

	if err != nil {
		return err
	}

	NegFloat32Native(out, in)

	return nil
}

func runSqrtFloat32(args ...tensor.Tensor) error {
	in, out, err := unaryFloat32Args(args)

	if err != nil {
		return err
	}

	SqrtFloat32Native(out, in)

	return nil
}

func init() {
	// abs, neg, sqrt: SIMD-specialized runners route through the per-arch
	// dispatcher into NEON / AVX-512 / AVX2 / SSE2 paths.
	registerUnarySIMD("abs", runAbsFloat32)
	registerUnarySIMD("neg", runNegFloat32)
	registerUnarySIMD("sqrt", runSqrtFloat32)
	registerUnarySIMDBFloat16("abs", runAbsBFloat16)
	registerUnarySIMDBFloat16("neg", runNegBFloat16)
	registerUnarySIMDBFloat16("sqrt", runSqrtBFloat16)
	registerUnarySIMDFloat16("abs", runAbsFloat16)
	registerUnarySIMDFloat16("neg", runNegFloat16)
	registerUnarySIMDFloat16("sqrt", runSqrtFloat16)

	registerUnary("square", func(value float32) float32 { return value * value })
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
