//go:build !amd64 && !arm64

package math

import gomath "math"

func reduceSum(input []float64) float64 {
	sum := 0.0

	for _, value := range input {
		sum += value
	}

	return sum
}

func reduceMax(input []float64) float64 {
	if len(input) == 0 {
		return -gomath.MaxFloat64
	}

	maxValue := input[0]

	for _, value := range input[1:] {
		if value > maxValue {
			maxValue = value
		}
	}

	return maxValue
}

func divScalar(output []float64, scalar float64) {
	for index := range output {
		output[index] /= scalar
	}
}

func addVec(output, left, right []float64) {
	for index := range left {
		output[index] = left[index] + right[index]
	}
}

func mulVec(output, left, right []float64) {
	for index := range left {
		output[index] = left[index] * right[index]
	}
}

func mulScalar(output []float64, scalar float64) {
	for index := range output {
		output[index] *= scalar
	}
}

func reduceSumSq(input []float64) float64 {
	sum := 0.0

	for _, value := range input {
		sum += value * value
	}

	return sum
}

func signVec(output, input []float64) {
	for index, value := range input {
		switch {
		case value > 0:
			output[index] = 1
		case value < 0:
			output[index] = -1
		default:
			output[index] = 0
		}
	}
}

func outerRow(output, input []float64, scale float64) {
	for index, value := range input {
		output[index] = scale * value
	}
}

func addScaledVec(output, input []float64, scale float64) {
	for index, value := range input {
		output[index] += scale * value
	}
}

func sqrtVec(output, input []float64) {
	for index, value := range input {
		output[index] = gomath.Sqrt(value)
	}
}

func addScalarVec(output []float64, scalar float64) {
	for index := range output {
		output[index] += scalar
	}
}

func divVec(output, left, right []float64) {
	for index := range left {
		output[index] = left[index] / right[index]
	}
}

func l2NormSq(input []float64) float64 {
	return reduceSumSq(input)
}

func expVec(output, input []float64) {
	for index, value := range input {
		output[index] = gomath.Exp(value)
	}
}

func logVec(output, input []float64) {
	for index, value := range input {
		output[index] = gomath.Log(value)
	}
}

func clampVec(output []float64, low, high float64) {
	for index, value := range output {
		if value < low {
			output[index] = low
			continue
		}

		if value > high {
			output[index] = high
		}
	}
}
