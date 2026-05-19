package neon

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
	// Use NEON f64-accumulator sum (existing SumFloat32Native) for
	// precision; divide by n on the scalar side.
	return float64(SumFloat32Native(row)) / float64(len(row))
}

func computeRowVariance(row []float32, mean float64) float64 {
	// NEON squared-diff sum, then divide by n.
	return float64(LayerNormSquaredDiffSumNative(row, float32(mean))) / float64(len(row))
}

func applyRowNormalization(
	row, outRow, scale, bias []float32,
	mean, invStdDev float64,
) {
	LayerNormApplyRowNative(outRow, row, scale, bias, float32(mean), float32(invStdDev))
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
	// Sum of squares via the NEON dot kernel (f64 accumulation
	// internally) — single-pass and ~64 GB/s for the f32 path.
	sumOfSquares := DotFloat32Native(row, row)
	meanSquare := float64(sumOfSquares) / float64(len(row))
	invRMS := 1.0 / math.Sqrt(meanSquare+rmsNormEpsilon)
	invRMSf32 := float32(invRMS)

	// Vectorized output: outRow[i] = row[i] * (invRMS * scale[i]).
	// Borrow a scratch buffer for the combined factor so the inner
	// multiply runs through MulFloat32Native (NEON FMUL .4S).
	combined := BorrowFloat32Buffer(len(row))
	defer ReleaseFloat32Buffer(combined)

	for index := range scale {
		combined[index] = invRMSf32 * scale[index]
	}

	MulFloat32Native(outRow, row, combined)
}
