//go:build !amd64 && !arm64

package math

import gomath "math"

func logSumExpKernel(out, input []float64, dimSize int) {
	for row := range out {
		inputRow := input[row*dimSize : (row+1)*dimSize]
		maxValue := inputRow[0]

		for _, value := range inputRow[1:] {
			if value > maxValue {
				maxValue = value
			}
		}

		sum := 0.0

		for _, value := range inputRow {
			sum += gomath.Exp(value - maxValue)
		}

		out[row] = maxValue + gomath.Log(sum)
	}
}
