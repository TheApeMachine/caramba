//go:build arm64

package neon

//go:noescape
func ConvTranspose2dTapNEONAsm(
	outRow *float32,
	weightVal float32,
	inputCol *float32,
	outCols int,
)

func ConvTranspose2dStride1RowNEON(
	outputRow, inputChannel, weightBlock []float32,
	outCols, kernelHeight, kernelWidth, inHeight, inWidth int,
	outRowIndex, blockStartCol int,
) {
	blockCols := outCols &^ 3
	if blockCols == 0 {
		return
	}

	for kernelRow := range kernelHeight {
		inputRow := outRowIndex - kernelRow
		if inputRow < 0 || inputRow >= inHeight {
			continue
		}

		inputRowOffset := inputRow * inWidth
		weightRowOffset := kernelRow * kernelWidth

		for kernelCol := range kernelWidth {
			inputCol := blockStartCol - kernelCol
			if inputCol < 0 || inputCol+blockCols > inWidth {
				continue
			}

			weightVal := weightBlock[weightRowOffset+kernelCol]

			ConvTranspose2dTapNEONAsm(
				&outputRow[0],
				weightVal,
				&inputChannel[inputRowOffset+inputCol],
				blockCols,
			)
		}
	}
}
