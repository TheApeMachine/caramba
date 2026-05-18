package kernels

import (
	"math"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
	"github.com/theapemachine/caramba/pkg/dtype"
)

/*
Softmax kernel. Operates along the last dimension. Uses the
numerically stable form: subtract max, exponentiate, normalize.

The body decomposes into findRowMax, fillShiftedExps, and
normalizeRow so the top-level runSoftmaxFloat32 stays under the
30-line method cap and each phase is independently testable.
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

		maximum := findRowMax(row)
		sum := fillShiftedExps(row, outRow, maximum)
		normalizeRow(outRow, sum)
	}

	return nil
}

/*
findRowMax returns the maximum value in the row.
*/
func findRowMax(row []float32) float32 {
	maximum := row[0]

	for _, candidate := range row[1:] {
		if candidate > maximum {
			maximum = candidate
		}
	}

	return maximum
}

/*
fillShiftedExps computes outRow[i] = exp(row[i] - maximum) and
returns the sum across the row. The shift-by-max form keeps the
exponent argument non-positive and prevents overflow.
*/
func fillShiftedExps(row []float32, outRow []float32, maximum float32) float32 {
	var sum float32

	for index, candidate := range row {
		shifted := float32(math.Exp(float64(candidate - maximum)))
		outRow[index] = shifted
		sum += shifted
	}

	return sum
}

/*
normalizeRow divides each entry by sum. When sum is zero (every
shifted exp underflowed — the row is fully masked or padded), the
row is left as all-zeros which yields a zero attention output for
this query position. This mirrors the attention kernel's
zero-denominator behavior; see attention.go for the rationale.
*/
func normalizeRow(outRow []float32, sum float32) {
	if sum == 0 {
		return
	}

	for index := range outRow {
		outRow[index] /= sum
	}
}
