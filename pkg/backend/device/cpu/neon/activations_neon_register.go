package neon

import (
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Real NEON registrations for exp, sigmoid, silu, swish, tanh — these
replace the per-lane scalar drivers (registerUnary skips these names
to avoid a duplicate registration). The f32 paths call into the
hand-written NEON assembly; bf16/fp16 widen-route-narrow through the
same f32 backend.
*/

func init() {
	registerNEONUnary("exp", ExpFloat32Native)
	registerNEONUnary("sigmoid", SigmoidFloat32Native)
	registerNEONUnary("silu", SiluFloat32Native)
	registerNEONUnary("swish", SiluFloat32Native) // swish = silu with β=1
	registerNEONUnary("tanh", TanhFloat32Native)
	registerNEONUnary("log", LogFloat32Native)
	registerNEONUnary("gelu_tanh", GeluTanhFloat32Native)
}

func registerNEONUnary(name string, native func(dst, src []float32)) {
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

			native(out, in)
			return nil
		},
	})

	// bf16: widen via NEON, run NEON activation, narrow via NEON.
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

			scratch := BorrowFloat32Buffer(len(in))
			outF32 := BorrowFloat32Buffer(len(in))
			defer ReleaseFloat32Buffer(scratch)
			defer ReleaseFloat32Buffer(outF32)

			Bfloat16BulkToFloat32(scratch, in)
			native(outF32, scratch)
			Float32BulkToBFloat16(out, outF32)
			return nil
		},
	})

	// fp16: same pattern with fp16 conversion.
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

			scratch := BorrowFloat32Buffer(len(in))
			outF32 := BorrowFloat32Buffer(len(in))
			defer ReleaseFloat32Buffer(scratch)
			defer ReleaseFloat32Buffer(outF32)

			Float16BulkToFloat32(scratch, in)
			native(outF32, scratch)
			Float32BulkToFloat16(out, outF32)
			return nil
		},
	})
}
