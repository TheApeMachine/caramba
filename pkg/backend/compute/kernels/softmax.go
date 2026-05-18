package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Softmax kernel. Operates along the last dimension. Uses the
numerically stable form: subtract max, exponentiate, normalize.
*/

func init() {
	Default.Register(Kernel{
		Name: "softmax",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runSoftmaxFloat32,
	})
}

func runSoftmaxFloat32(args ...tensor.Tensor) error {
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

	dims := args[0].Shape().Dims()

	if len(dims) == 0 {
		return tensor.ErrShapeMismatch
	}

	lastDim := dims[len(dims)-1]
	rows := len(input) / lastDim

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := out[rowIndex*lastDim : (rowIndex+1)*lastDim]

		maximum := row[0]

		for _, value := range row[1:] {
			if value > maximum {
				maximum = value
			}
		}

		sum := float32(0)

		for index, value := range row {
			shifted := math.Exp(float64(value - maximum))
			outRow[index] = float32(shifted)
			sum += float32(shifted)
		}

		if sum == 0 {
			continue
		}

		for index := range outRow {
			outRow[index] /= sum
		}
	}

	return nil
}
