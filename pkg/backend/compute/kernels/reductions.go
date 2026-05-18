package kernels

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
}

func init() {
	registerReduce("sum", func(values []float32) float32 {
		var sum float64

		for _, value := range values {
			sum += float64(value)
		}

		return float32(sum)
	})

	registerReduce("mean", func(values []float32) float32 {
		var sum float64

		for _, value := range values {
			sum += float64(value)
		}

		return float32(sum / float64(len(values)))
	})

	registerReduce("prod", func(values []float32) float32 {
		product := float64(1)

		for _, value := range values {
			product *= float64(value)
		}

		return float32(product)
	})

	registerReduce("reduce_min", func(values []float32) float32 {
		minimum := values[0]

		for _, value := range values[1:] {
			if value < minimum {
				minimum = value
			}
		}

		return minimum
	})

	registerReduce("reduce_max", func(values []float32) float32 {
		maximum := values[0]

		for _, value := range values[1:] {
			if value > maximum {
				maximum = value
			}
		}

		return maximum
	})

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

	registerReduce("l1_norm", func(values []float32) float32 {
		var sum float64

		for _, value := range values {
			sum += math.Abs(float64(value))
		}

		return float32(sum)
	})

	registerReduce("l2_norm", func(values []float32) float32 {
		var sum float64

		for _, value := range values {
			sum += float64(value) * float64(value)
		}

		return float32(math.Sqrt(sum))
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
