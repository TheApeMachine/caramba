package math

import "math"

/*
InvSqrtDimScaleGeneric writes out[i] = input[i] / sqrt(dim) for dim > 0.
*/
func InvSqrtDimScaleGeneric(out, input []float32, dim int32) {
	if dim <= 0 || len(input) != len(out) {
		return
	}

	scale := float32(1.0 / math.Sqrt(float64(dim)))

	for index, value := range input {
		out[index] = value * scale
	}
}

/*
LogSumExpRowGeneric computes log-sum-exp for one row of length cols.
*/
func LogSumExpRowGeneric(row []float32) float32 {
	if len(row) == 0 {
		return 0
	}

	maximum := row[0]

	for _, candidate := range row[1:] {
		if candidate > maximum {
			maximum = candidate
		}
	}

	var accumulator float64

	for _, candidate := range row {
		accumulator += math.Exp(float64(candidate - maximum))
	}

	return maximum + float32(math.Log(accumulator))
}

/*
LogSumExpGeneric reduces the last dimension of a row-major input.
*/
func LogSumExpGeneric(input []float32, cols int, out []float32) {
	if cols <= 0 || len(input)%cols != 0 || len(out) != len(input)/cols {
		return
	}

	rows := len(input) / cols

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowOffset := rowIndex * cols
		out[rowIndex] = LogSumExpRowGeneric(input[rowOffset : rowOffset+cols])
	}
}

/*
OuterGeneric computes out[i,j] = left[i] * right[j] in row-major layout.
*/
func OuterGeneric(left, right, out []float32) {
	if len(out) != len(left)*len(right) {
		return
	}

	rightLen := len(right)

	for leftIndex, leftValue := range left {
		rowOffset := leftIndex * rightLen

		for rightIndex, rightValue := range right {
			out[rowOffset+rightIndex] = leftValue * rightValue
		}
	}
}
