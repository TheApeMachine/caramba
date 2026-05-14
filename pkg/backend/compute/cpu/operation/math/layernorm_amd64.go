//go:build amd64

package math

func layerNormKernel(out, input, weight, bias []float64, eps float64, dimSize int) {
	for row := 0; row < len(input)/dimSize; row++ {
		inputRow := input[row*dimSize : (row+1)*dimSize]
		outputRow := out[row*dimSize : (row+1)*dimSize]

		if useAVX2 && useFMA {
			layerNormRowAVX2(outputRow, inputRow, weight, bias, eps)

			continue
		}

		layerNormRowSSE2(outputRow, inputRow, weight, bias, eps)
	}
}
