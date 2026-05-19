package neon

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Reduction kernels. The default signature reduces the entire input to
a single scalar value; axis-specific reductions land via the
*AlongAxis helpers in a follow-up. The args order is (input, output)
with output's shape being [1] for the global variant.
*/

type reducer func(values []float32) float32

func reduceFloat32(op reducer) func(args ...tensor.Tensor) error {
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

		if len(out) < 1 || len(input) == 0 {
			return tensor.ErrShapeMismatch
		}

		out[0] = op(input)
		return nil
	}
}

func registerReduce(name string, op reducer) {
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       reduceFloat32(op),
	})

	// Auto-register bf16 and fp16 variants via widen→f32-reduce→narrow.
	// The "sum" reduction is registered separately with a NEON-vectorized
	// dtype-aware path; this generic auto-register is skipped for it.
	if name == "sum" {
		return
	}

	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.BFloat16},
			Outputs: []dtype.DType{dtype.BFloat16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       reduceBFloat16(op),
	})
	Default.Register(Kernel{
		Name: name,
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float16},
			Outputs: []dtype.DType{dtype.Float16},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       reduceFloat16Generic(op),
	})
}

func reduceBFloat16(op reducer) func(args ...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 2 {
			return tensor.ErrShapeMismatch
		}

		input, err := args[0].BFloat16Native()

		if err != nil {
			return err
		}

		out, err := args[1].BFloat16Native()

		if err != nil {
			return err
		}

		if len(out) < 1 || len(input) == 0 {
			return tensor.ErrShapeMismatch
		}

		widened := BorrowFloat32Buffer(len(input))
		defer ReleaseFloat32Buffer(widened)
		Bfloat16BulkToFloat32(widened, input)

		out[0] = dtype.NewBfloat16FromFloat32(op(widened))
		return nil
	}
}

func reduceFloat16Generic(op reducer) func(args ...tensor.Tensor) error {
	return func(args ...tensor.Tensor) error {
		if len(args) != 2 {
			return tensor.ErrShapeMismatch
		}

		input, err := args[0].Float16Native()

		if err != nil {
			return err
		}

		out, err := args[1].Float16Native()

		if err != nil {
			return err
		}

		if len(out) < 1 || len(input) == 0 {
			return tensor.ErrShapeMismatch
		}

		widened := BorrowFloat32Buffer(len(input))
		defer ReleaseFloat32Buffer(widened)
		Float16BulkToFloat32(widened, input)

		out[0] = dtype.Fromfloat32(op(widened))
		return nil
	}
}

func registerReduceBFloat16(name string, run func(args ...tensor.Tensor) error) {
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

func registerReduceFloat16(name string, run func(args ...tensor.Tensor) error) {
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

func runSumFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].Float16Native()

	if err != nil {
		return err
	}

	out, err := args[1].Float16Native()

	if err != nil {
		return err
	}

	if len(out) < 1 || len(input) == 0 {
		return tensor.ErrShapeMismatch
	}

	out[0] = SumFloat16Native(input)
	return nil
}

func runSumBFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].BFloat16Native()

	if err != nil {
		return err
	}

	out, err := args[1].BFloat16Native()

	if err != nil {
		return err
	}

	if len(out) < 1 || len(input) == 0 {
		return tensor.ErrShapeMismatch
	}

	out[0] = SumBFloat16Native(input)
	return nil
}

func init() {
	registerReduce("sum", SumFloat32Native)
	registerReduceBFloat16("sum", runSumBFloat16)
	registerReduceFloat16("sum", runSumFloat16)

	registerReduce("mean", func(values []float32) float32 {
		return SumFloat32Native(values) / float32(len(values))
	})

	registerReduce("prod", ReduceProdFloat32Native)

	registerReduce("reduce_min", ReduceMinFloat32Native)

	registerReduce("reduce_max", ReduceMaxFloat32Native)

	registerReduce("argmin", func(values []float32) float32 {
		index := 0
		minimum := values[0]

		for cursor, value := range values[1:] {
			if value < minimum {
				minimum = value
				index = cursor + 1
			}
		}

		return float32(index)
	})

	registerReduce("argmax", func(values []float32) float32 {
		index := 0
		maximum := values[0]

		for cursor, value := range values[1:] {
			if value > maximum {
				maximum = value
				index = cursor + 1
			}
		}

		return float32(index)
	})

	registerReduce("l1_norm", L1NormFloat32Native)

	registerReduce("l2_norm", func(values []float32) float32 {
		return float32(math.Sqrt(float64(DotFloat32Native(values, values))))
	})

	registerReduce("variance", func(values []float32) float32 {
		var sum float64

		for _, value := range values {
			sum += float64(value)
		}

		mean := sum / float64(len(values))
		var variance float64

		for _, value := range values {
			delta := float64(value) - mean
			variance += delta * delta
		}

		return float32(variance / float64(len(values)))
	})

	registerReduce("stddev", func(values []float32) float32 {
		var sum float64

		for _, value := range values {
			sum += float64(value)
		}

		mean := sum / float64(len(values))
		var variance float64

		for _, value := range values {
			delta := float64(value) - mean
			variance += delta * delta
		}

		return float32(math.Sqrt(variance / float64(len(values))))
	})
}
