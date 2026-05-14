//go:build amd64

package math

func logSumExpKernel(out, input []float64, dimSize int) {
	for row := range out {
		inputRow := input[row*dimSize : (row+1)*dimSize]

		if useAVX2 && useFMA {
			out[row] = logSumExpRowAVX2(inputRow)

			continue
		}

		out[row] = logSumExpRowSSE2(inputRow)
	}
}
