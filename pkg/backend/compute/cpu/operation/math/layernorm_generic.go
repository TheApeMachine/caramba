//go:build !amd64 && !arm64

package math

import gomath "math"

func layerNormKernel(out, input, weight, bias []float64, eps float64, dimSize int) {
	for row := 0; row < len(input)/dimSize; row++ {
		inputRow := input[row*dimSize : (row+1)*dimSize]
		outputRow := out[row*dimSize : (row+1)*dimSize]
		mean := 0.0

		for _, value := range inputRow {
			mean += value
		}

		mean /= float64(dimSize)
		variance := 0.0

		for _, value := range inputRow {
			delta := value - mean
			variance += delta * delta
		}

		scale := 1 / gomath.Sqrt(variance/float64(dimSize)+eps)

		for index, value := range inputRow {
			outputRow[index] = (value-mean)*scale*weight[index] + bias[index]
		}
	}
}
