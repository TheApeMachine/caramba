//go:build !amd64 && !arm64

package math

import gomath "math"

func softmaxKernel(out, input []float64, dimSize int) {
	for row := 0; row < len(input)/dimSize; row++ {
		inputRow := input[row*dimSize : (row+1)*dimSize]
		outputRow := out[row*dimSize : (row+1)*dimSize]
		maxValue := inputRow[0]

		for _, value := range inputRow[1:] {
			if value > maxValue {
				maxValue = value
			}
		}

		sum := 0.0

		for index, value := range inputRow {
			expValue := gomath.Exp(value - maxValue)
			outputRow[index] = expValue
			sum += expValue
		}

		for index := range outputRow {
			outputRow[index] /= sum
		}
	}
}
