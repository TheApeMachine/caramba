package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
LayerNorm and RMSNorm kernels. Both operate along the last dimension
with optional scale and bias.

Body decomposed into computeRowMean, computeRowVariance, and
applyRowNormalization so the top-level driver stays small and each
phase is independently testable.

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

const layerNormEpsilon = 1e-5
const rmsNormEpsilon = 1e-6

func runLayerNormFloat32(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	input, scale, bias, out, lastDim, err := layerNormViews(args)

	if err != nil {
		return err
	}

	rows := len(input) / lastDim

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := out[rowIndex*lastDim : (rowIndex+1)*lastDim]

		mean := computeRowMean(row)
		variance := computeRowVariance(row, mean)
		invStdDev := 1.0 / math.Sqrt(variance+layerNormEpsilon)
		applyRowNormalization(row, outRow, scale, bias, mean, invStdDev)
	}

	return nil
}

func layerNormViews(args []tensor.Tensor) (input, scale, bias, out []float32, lastDim int, err error) {
	input, err = args[0].Float32Native()

	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	scale, err = args[1].Float32Native()

	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	bias, err = args[2].Float32Native()

	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	out, err = args[3].Float32Native()

	if err != nil {
		return nil, nil, nil, nil, 0, err
	}

	dims := args[0].Shape().Dims()

	if len(dims) == 0 {
		return nil, nil, nil, nil, 0, tensor.ErrShapeMismatch
	}

	lastDim = dims[len(dims)-1]

	if len(scale) != lastDim || len(bias) != lastDim || len(out) != len(input) {
		return nil, nil, nil, nil, 0, tensor.ErrShapeMismatch
	}

	return input, scale, bias, out, lastDim, nil
}

func computeRowMean(row []float32) float64 {
	var sum float64

	for _, value := range row {
		sum += float64(value)
	}

	return sum / float64(len(row))
}

func computeRowVariance(row []float32, mean float64) float64 {
	var variance float64

	for _, value := range row {
		delta := float64(value) - mean
		variance += delta * delta
	}

	return variance / float64(len(row))
}

func applyRowNormalization(
	row, outRow, scale, bias []float32,
	mean, invStdDev float64,
) {
	for index, value := range row {
		normalized := (float64(value) - mean) * invStdDev
		outRow[index] = float32(normalized)*scale[index] + bias[index]
	}
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

	rows := len(input) / lastDim

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := out[rowIndex*lastDim : (rowIndex+1)*lastDim]
		applyRMSRow(row, outRow, scale)
	}

	return nil
}

func applyRMSRow(row, outRow, scale []float32) {
	var meanSquare float64

	for _, value := range row {
		meanSquare += float64(value) * float64(value)
	}

	meanSquare /= float64(len(row))
	invRMS := 1.0 / math.Sqrt(meanSquare+rmsNormEpsilon)

	for index, value := range row {
		outRow[index] = float32(float64(value)*invRMS) * scale[index]
	}
}
