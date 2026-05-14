//go:build amd64

package math

func softmaxKernel(out, input []float64, dimSize int) {
	copy(out, input)

	for row := 0; row < len(out)/dimSize; row++ {
		outputRow := out[row*dimSize : (row+1)*dimSize]

		if useAVX2 && useFMA {
			softmaxRowAVX2(outputRow)

			continue
		}

		softmaxRowSSE2(outputRow)
	}
}
