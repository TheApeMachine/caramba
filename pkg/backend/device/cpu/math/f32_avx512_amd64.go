//go:build amd64

package math

import "math"

//go:noescape
func InvSqrtDimScaleFloat32AVX512Asm(out, input *float32, scale float32, count int)

//go:noescape
func LogSumExpRowPartsFloat32AVX512Asm(row *float32, cols int, maximum, expSum *float32)

//go:noescape
func OuterFloat32AVX512Asm(out, left, right *float32, leftCount, rightCount int)

func InvSqrtDimScaleF32AVX512(out, input []float32, dim int32) {
	if len(out) == 0 {
		return
	}

	scale := float32(1.0 / math.Sqrt(float64(dim)))
	InvSqrtDimScaleFloat32AVX512Asm(&out[0], &input[0], scale, len(out))
}

func logSumExpRowF32AVX512(row []float32) float32 {
	if len(row) == 0 {
		return 0
	}

	var maximum float32
	var expSum float32

	LogSumExpRowPartsFloat32AVX512Asm(&row[0], len(row), &maximum, &expSum)

	return maximum + float32(math.Log(float64(expSum)))
}

func LogSumExpF32AVX512(input []float32, cols int, out []float32) {
	if cols <= 0 || len(input)%cols != 0 {
		return
	}

	rows := len(input) / cols

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowOffset := rowIndex * cols
		out[rowIndex] = logSumExpRowF32AVX512(input[rowOffset : rowOffset+cols])
	}
}

func OuterF32AVX512(left, right, out []float32) {
	if len(left) == 0 || len(right) == 0 {
		return
	}

	OuterFloat32AVX512Asm(&out[0], &left[0], &right[0], len(left), len(right))
}
