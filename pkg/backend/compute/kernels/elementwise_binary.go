package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Elementwise binary kernels: two same-shape tensors in → one out.
The dispatcher routes (a, b, out). Broadcasting lives one layer up;
these kernels require matching shapes.
*/

type binaryOp func(a, b float32) float32

func runDivFloat32(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	left, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	right, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return tensor.ErrShapeMismatch
	}

	divFloat32Native(out, left, right)

	return nil
}

func binaryFloat32(op binaryOp) func(args ...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 3 {
			return tensor.ErrShapeMismatch
		}

		left, err := args[0].Float32Native()

		if err != nil {
			return err
		}

		right, err := args[1].Float32Native()

		if err != nil {
			return err
		}

		out, err := args[2].Float32Native()

		if err != nil {
			return err
		}

		if len(left) != len(right) || len(out) != len(left) {
			return tensor.ErrShapeMismatch
		}

		for index, value := range left {
			out[index] = op(value, right[index])
		}

		return nil
	}
}

func registerBinary(name string, op binaryOp) {
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       binaryFloat32(op),
	})
}

func init() {
	// div uses a SIMD-specialized runner so the dispatcher per architecture
	// can route into NEON / AVX-512 / AVX2 / SSE2 paths.
	Default.Register(Kernel{
		Name: "div",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runDivFloat32,
	})
	registerBinary("pow", func(a, b float32) float32 {
		return float32(math.Pow(float64(a), float64(b)))
	})
	registerBinary("atan2", func(a, b float32) float32 {
		return float32(math.Atan2(float64(a), float64(b)))
	})
	registerBinary("max", func(a, b float32) float32 {
		if a > b {
			return a
		}

		return b
	})
	registerBinary("min", func(a, b float32) float32 {
		if a < b {
			return a
		}

		return b
	})
	registerBinary("mod", func(a, b float32) float32 {
		return float32(math.Mod(float64(a), float64(b)))
	})

	// Comparison kernels: output 0/1 in float32 so they slot into the
	// same dispatcher; integer-valued mask producers live in their
	// own signatures.
	registerBinary("eq", func(a, b float32) float32 {
		if a == b {
			return 1
		}
		return 0
	})
	registerBinary("ne", func(a, b float32) float32 {
		if a != b {
			return 1
		}
		return 0
	})
	registerBinary("lt", func(a, b float32) float32 {
		if a < b {
			return 1
		}
		return 0
	})
	registerBinary("le", func(a, b float32) float32 {
		if a <= b {
			return 1
		}
		return 0
	})
	registerBinary("gt", func(a, b float32) float32 {
		if a > b {
			return 1
		}
		return 0
	})
	registerBinary("ge", func(a, b float32) float32 {
		if a >= b {
			return 1
		}
		return 0
	})
}
