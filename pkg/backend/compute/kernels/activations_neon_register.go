package kernels

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
	registerNEONUnary("exp", expFloat32Native)
	registerNEONUnary("sigmoid", sigmoidFloat32Native)
	registerNEONUnary("silu", siluFloat32Native)
	registerNEONUnary("swish", siluFloat32Native) // swish = silu with β=1
	registerNEONUnary("tanh", tanhFloat32Native)
	registerNEONUnary("log", logFloat32Native)
	registerNEONUnary("gelu_tanh", geluTanhFloat32Native)
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

			scratch := borrowFloat32Buffer(len(in))
			outF32 := borrowFloat32Buffer(len(in))
			defer releaseFloat32Buffer(scratch)
			defer releaseFloat32Buffer(outF32)

			bfloat16BulkToFloat32(scratch, in)
			native(outF32, scratch)
			float32BulkToBFloat16(out, outF32)
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

			scratch := borrowFloat32Buffer(len(in))
			outF32 := borrowFloat32Buffer(len(in))
			defer releaseFloat32Buffer(scratch)
			defer releaseFloat32Buffer(outF32)

			float16BulkToFloat32(scratch, in)
			native(outF32, scratch)
			float32BulkToFloat16(out, outF32)
			return nil
		},
	})
}
