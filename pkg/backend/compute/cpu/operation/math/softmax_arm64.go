//go:build arm64

package math

func softmaxKernel(out, input []float64, dimSize int) {
	copy(out, input)

	for row := 0; row < len(out)/dimSize; row++ {
		softmaxRowNEON(out[row*dimSize : (row+1)*dimSize])
	}
}
