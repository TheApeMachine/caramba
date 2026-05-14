//go:build arm64

package math

func logSumExpKernel(out, input []float64, dimSize int) {
	for row := range out {
		inputRow := input[row*dimSize : (row+1)*dimSize]
		out[row] = logSumExpRowNEON(inputRow)
	}
}
