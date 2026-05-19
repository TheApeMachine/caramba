package neon

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/activation"
	"github.com/theapemachine/caramba/pkg/dtype"
)

func init() {
	registerNEONActivations([]struct {
		name string
		run  func(dst, src unsafe.Pointer, count int, format dtype.DType)
	}{
		{"exp", activation.Exp},
		{"log", activation.Log},
		{"log1p", activation.Log1p},
		{"expm1", activation.Expm1},
		{"sigmoid", activation.Sigmoid},
		{"log_sigmoid", activation.LogSigmoid},
		{"tanh", activation.Tanh},
		{"silu", activation.Silu},
		{"swish", activation.Swish},
		{"gelu_tanh", activation.GeluTanh},
		{"gelu", activation.Gelu},
		{"relu", activation.ReLU},
		{"leaky_relu", activation.LeakyReLU},
		{"elu", activation.ELU},
		{"celu", activation.CELU},
		{"selu", activation.SELU},
		{"softplus", activation.Softplus},
		{"mish", activation.Mish},
		{"softsign", activation.Softsign},
		{"hardsigmoid", activation.HardSigmoid},
		{"hardswish", activation.HardSwish},
		{"hardtanh", activation.HardTanh},
		{"hardgelu", activation.HardGelu},
		{"quickgelu", activation.QuickGelu},
		{"tanhshrink", activation.TanhShrink},
	})

	registerNEONSoftmax("softmax", activation.Softmax)
	registerNEONSoftmax("log_softmax", activation.LogSoftmax)

	registerNEONPackedGated("swiglu", activation.SwiGLU)
	registerNEONPackedGated("glu", activation.GLU)
	registerNEONPackedGated("geglu", activation.GeGLU)
	registerNEONPackedGated("geglu_tanh", activation.GeGLUTanh)
	registerNEONPackedGated("reglu", activation.ReGLU)
	registerNEONPackedGated("siglu", activation.SiGLU)
	registerNEONPackedGated("linglu", activation.LinGLU)
	registerNEONPackedGated("seglu", activation.SeGLU)
}

func registerNEONSoftmax(
	name string,
	run func(dst, src unsafe.Pointer, count int, format dtype.DType),
) {
	registerNEONUnary(name, run)
}

func registerNEONActivations(specs []struct {
	name string
	run  func(dst, src unsafe.Pointer, count int, format dtype.DType)
}) {
	for _, spec := range specs {
		registerNEONUnary(spec.name, spec.run)
	}
}

func registerNEONPackedGated(
	name string,
	run func(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType),
) {
	registerPackedGatedKernel(name, dtype.Float32, run)
	registerPackedGatedKernel(name, dtype.BFloat16, run)
	registerPackedGatedKernel(name, dtype.Float16, run)
}

func registerPackedGatedKernel(
	name string,
	storage dtype.DType,
	run func(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType),
) {
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storage},
			Outputs: []dtype.DType{storage},
		},
		Locations: []tensor.Location{tensor.Host},
		Run: func(args ...tensor.Tensor) error {
			if len(args) != 2 {
				return tensor.ErrShapeMismatch
			}

			batch, halfCount, ok := activation.PackedGatedShape(args[0].Shape())
			if !ok {
				return tensor.ErrShapeMismatch
			}

			expectedOut := batch * halfCount
			if args[1].Shape().Len() != expectedOut {
				return tensor.ErrShapeMismatch
			}

			switch storage {
			case dtype.Float32:
				in, err := args[0].Float32Native()
				if err != nil {
					return err
				}

				out, err := args[1].Float32Native()
				if err != nil {
					return err
				}

				run(
					unsafe.Pointer(unsafe.SliceData(out)),
					unsafe.Pointer(unsafe.SliceData(in)),
					batch,
					halfCount,
					storage,
				)
			case dtype.BFloat16:
				in, err := args[0].BFloat16Native()
				if err != nil {
					return err
				}

				out, err := args[1].BFloat16Native()
				if err != nil {
					return err
				}

				run(
					unsafe.Pointer(unsafe.SliceData(out)),
					unsafe.Pointer(unsafe.SliceData(in)),
					batch,
					halfCount,
					storage,
				)
			case dtype.Float16:
				in, err := args[0].Float16Native()
				if err != nil {
					return err
				}

				out, err := args[1].Float16Native()
				if err != nil {
					return err
				}

				run(
					unsafe.Pointer(unsafe.SliceData(out)),
					unsafe.Pointer(unsafe.SliceData(in)),
					batch,
					halfCount,
					storage,
				)
			}

			return nil
		},
	})
}

func registerNEONUnary(
	name string,
	run func(dst, src unsafe.Pointer, count int, format dtype.DType),
) {
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run: func(args ...tensor.Tensor) error {
			if len(args) != 2 {
				return tensor.ErrShapeMismatch
			}

			in, err := args[0].Float32Native()
			if err != nil {
				return err
			}

			out, err := args[1].Float32Native()
			if err != nil {
				return err
			}

			if len(in) != len(out) {
				return tensor.ErrShapeMismatch
			}

			run(
				unsafe.Pointer(unsafe.SliceData(out)),
				unsafe.Pointer(unsafe.SliceData(in)),
				len(in),
				dtype.Float32,
			)
			return nil
		},
	})

	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run: func(args ...tensor.Tensor) error {
			if len(args) != 2 {
				return tensor.ErrShapeMismatch
			}

			in, err := args[0].BFloat16Native()
			if err != nil {
				return err
			}

			out, err := args[1].BFloat16Native()
			if err != nil {
				return err
			}

			if len(in) != len(out) {
				return tensor.ErrShapeMismatch
			}

			run(
				unsafe.Pointer(unsafe.SliceData(out)),
				unsafe.Pointer(unsafe.SliceData(in)),
				len(in),
				dtype.BFloat16,
			)
			return nil
		},
	})

	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run: func(args ...tensor.Tensor) error {
			if len(args) != 2 {
				return tensor.ErrShapeMismatch
			}

			in, err := args[0].Float16Native()
			if err != nil {
				return err
			}

			out, err := args[1].Float16Native()
			if err != nil {
				return err
			}

			if len(in) != len(out) {
				return tensor.ErrShapeMismatch
			}

			run(
				unsafe.Pointer(unsafe.SliceData(out)),
				unsafe.Pointer(unsafe.SliceData(in)),
				len(in),
				dtype.Float16,
			)
			return nil
		},
	})
}
