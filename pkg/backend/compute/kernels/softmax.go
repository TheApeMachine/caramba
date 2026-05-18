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
	registerSoftmaxKernel(dtype.Float32, runSoftmaxFloat32)
	registerSoftmaxKernel(dtype.Float16, runSoftmaxFloat16)
	registerSoftmaxKernel(dtype.BFloat16, runSoftmaxBFloat16)
}

func registerSoftmaxKernel(
	storageDType dtype.DType,
	run func(...tensor.Tensor) error,
) {
	Default.Register(Kernel{
		Name: "softmax",
		Signature: Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Host},
		Run:       run,
	})
}

func runSoftmaxFloat32(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	rows, lastDim, err := softmaxDims(args[0], args[1])
	if err != nil {
		return err
	}

	input, err := args[0].Float32Native()

	if err != nil {
		return err
	}

	out, err := args[1].Float32Native()

	if err != nil {
		return err
	}

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := out[rowIndex*lastDim : (rowIndex+1)*lastDim]

		maximum := findRowMax(row)
		sum := fillShiftedExps(row, outRow, maximum)
		normalizeRow(outRow, sum)
	}

	return nil
}

func runSoftmaxFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	rows, lastDim, err := softmaxDims(args[0], args[1])
	if err != nil {
		return err
	}

	input, err := args[0].Float16Native()
	if err != nil {
		return err
	}

	out, err := args[1].Float16Native()
	if err != nil {
		return err
	}

	scratch := make([]float32, lastDim)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := out[rowIndex*lastDim : (rowIndex+1)*lastDim]

		maximum := findRowMaxFloat16(row)
		sum := fillShiftedExpsFloat16(row, scratch, maximum)
		normalizeRowFloat16(outRow, scratch, sum)
	}

	return nil
}

func runSoftmaxBFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	rows, lastDim, err := softmaxDims(args[0], args[1])
	if err != nil {
		return err
	}

	input, err := args[0].BFloat16Native()
	if err != nil {
		return err
	}

	out, err := args[1].BFloat16Native()
	if err != nil {
		return err
	}

	scratch := make([]float32, lastDim)

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		row := input[rowIndex*lastDim : (rowIndex+1)*lastDim]
		outRow := out[rowIndex*lastDim : (rowIndex+1)*lastDim]

		maximum := findRowMaxBFloat16(row)
		sum := fillShiftedExpsBFloat16(row, scratch, maximum)
		normalizeRowBFloat16(outRow, scratch, sum)
	}

	return nil
}

func softmaxDims(input tensor.Tensor, out tensor.Tensor) (int, int, error) {
	if !input.Shape().Equal(out.Shape()) {
		return 0, 0, tensor.ErrShapeMismatch
	}

	dims := input.Shape().Dims()

	if len(dims) == 0 {
		return 0, 0, tensor.ErrShapeMismatch
	}

	lastDim := dims[len(dims)-1]
	if lastDim == 0 {
		return 0, 0, nil
	}

	return input.Len() / lastDim, lastDim, nil
}

/*
findRowMax returns the maximum value in the row.
*/
func findRowMax(row []float32) float32 {
	if len(row) == 0 {
		return 0
	}

	return reduceMaxFloat32Native(row)
}

func findRowMaxFloat16(row []dtype.F16) float32 {
	maximum := row[0].Float32()

	for _, candidate := range row[1:] {
		value := candidate.Float32()

		if value > maximum {
			maximum = value
		}
	}

	return maximum
}

func findRowMaxBFloat16(row []dtype.BF16) float32 {
	maximum := (&row[0]).Float32()

	for index := 1; index < len(row); index++ {
		value := (&row[index]).Float32()

		if value > maximum {
			maximum = value
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

func fillShiftedExpsFloat16(
	row []dtype.F16,
	scratch []float32,
	maximum float32,
) float32 {
	var sum float32

	for index, candidate := range row {
		shifted := float32(math.Exp(float64(candidate.Float32() - maximum)))
		scratch[index] = shifted
		sum += shifted
	}

	return sum
}

func fillShiftedExpsBFloat16(
	row []dtype.BF16,
	scratch []float32,
	maximum float32,
) float32 {
	var sum float32

	for index := range row {
		shifted := float32(math.Exp(float64((&row[index]).Float32() - maximum)))
		scratch[index] = shifted
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

func normalizeRowFloat16(outRow []dtype.F16, scratch []float32, sum float32) {
	if sum == 0 {
		return
	}

	for index, value := range scratch {
		outRow[index] = dtype.Fromfloat32(value / sum)
	}
}

func normalizeRowBFloat16(outRow []dtype.BF16, scratch []float32, sum float32) {
	if sum == 0 {
		return
	}

	for index, value := range scratch {
		outRow[index] = dtype.NewBfloat16FromFloat32(value / sum)
	}
}
