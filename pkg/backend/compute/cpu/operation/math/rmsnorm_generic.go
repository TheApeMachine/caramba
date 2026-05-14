//go:build !amd64 && !arm64

package math

import gomath "math"

func rmsNormKernel(out, input, weight []float64, eps float64, dimSize int) {
	for row := 0; row < len(input)/dimSize; row++ {
		inputRow := input[row*dimSize : (row+1)*dimSize]
		outputRow := out[row*dimSize : (row+1)*dimSize]
		sumSquares := 0.0

		for _, value := range inputRow {
			sumSquares += value * value
		}

		scale := 1 / gomath.Sqrt(sumSquares/float64(dimSize)+eps)

		for index, value := range inputRow {
			outputRow[index] = value * scale * weight[index]
		}
	}
}
