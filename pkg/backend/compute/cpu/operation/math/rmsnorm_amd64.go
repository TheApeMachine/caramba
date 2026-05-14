//go:build amd64

package math

func rmsNormKernel(out, input, weight []float64, eps float64, dimSize int) {
	for row := 0; row < len(input)/dimSize; row++ {
		inputRow := input[row*dimSize : (row+1)*dimSize]
		outputRow := out[row*dimSize : (row+1)*dimSize]

		if useAVX2 && useFMA {
			rmsNormRowAVX2(outputRow, inputRow, weight, eps)

			continue
		}

		rmsNormRowSSE2(outputRow, inputRow, weight, eps)
	}
}
