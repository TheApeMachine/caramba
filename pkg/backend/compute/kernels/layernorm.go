package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
LayerNorm and RMSNorm kernels. Both operate along the last dimension
with optional scale and bias. The signatures here register the basic
variant with mandatory scale; the bias parameter is implied as
optional and a separate signature lands when needed.

Args order for layernorm: (input, scale, bias, output).
Args order for rmsnorm: (input, scale, output).
*/

func init() {
	Default.Register(Kernel{
		Name: "layernorm",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runLayerNormFloat32,
	})

	Default.Register(Kernel{
		Name: "rmsnorm",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{dtype.Float32, dtype.Float32},
			Outputs: []dtype.DType{dtype.Float32},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       runRMSNormFloat32,
	})
}

func runLayerNormFloat32(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	scale, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	bias, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[3].Float32Native()

	if err != nil {
		return err
	}

	dims := args[0].Shape().Dims()

	if len(dims) == 0 {
		return tensor.ErrShapeMismatch
	}

	lastDim := dims[len(dims)-1]

	if len(scale) != lastDim || len(bias) != lastDim || len(out) != len(input) {
		return tensor.ErrShapeMismatch
	}

	const epsilon = 1e-5
	rows := len(input) / lastDim

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := out[rowIndex*lastDim : (rowIndex+1)*lastDim]

		var sum float64

		for _, value := range row {
			sum += float64(value)
		}

		mean := sum / float64(lastDim)

		var variance float64

		for _, value := range row {
			delta := float64(value) - mean
			variance += delta * delta
		}

		variance /= float64(lastDim)
		invStdDev := 1.0 / math.Sqrt(variance+epsilon)

		for index, value := range row {
			normalized := (float64(value) - mean) * invStdDev
			outRow[index] = float32(normalized)*scale[index] + bias[index]
		}
	}

	return nil
}

func runRMSNormFloat32(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	input, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	scale, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float32Native()

	if err != nil {
		return err
	}

	dims := args[0].Shape().Dims()

	if len(dims) == 0 {
		return tensor.ErrShapeMismatch
	}

	lastDim := dims[len(dims)-1]

	if len(scale) != lastDim || len(out) != len(input) {
		return tensor.ErrShapeMismatch
	}

	const epsilon = 1e-6
	rows := len(input) / lastDim

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := out[rowIndex*lastDim : (rowIndex+1)*lastDim]

		var meanSquare float64

		for _, value := range row {
			meanSquare += float64(value) * float64(value)
		}

		meanSquare /= float64(lastDim)
		invRMS := 1.0 / math.Sqrt(meanSquare+epsilon)

		for index, value := range row {
			outRow[index] = float32(float64(value)*invRMS) * scale[index]
		}
	}

	return nil
}
