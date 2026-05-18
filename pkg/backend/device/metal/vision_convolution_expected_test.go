package metal

import "github.com/theapemachine/caramba/pkg/dtype"

func conv1DDTypeBytes(width int, storageDType dtype.DType) ([]byte, []byte, []byte, []byte) {
	batch, inChannels, outChannels, kernelLength := 2, 2, 3, 3
	inputLength := width + kernelLength - 1
	inputValues := projectionValues(batch*inChannels*inputLength, 67, 64)
	weightValues := projectionValues(outChannels*inChannels*kernelLength, 37, 128)
	biasValues := projectionValues(outChannels, 17, 32)
	inputBytes := encodeProjectionValuesAsDType(inputValues, storageDType)
	weightBytes := encodeProjectionValuesAsDType(weightValues, storageDType)
	biasBytes := encodeProjectionValuesAsDType(biasValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	weightStored := decodeDTypeBytesToFloat32(weightBytes, storageDType)
	biasStored := decodeDTypeBytesToFloat32(biasBytes, storageDType)
	expected := conv1DExpectedValues(
		inputStored, weightStored, biasStored,
		batch, inChannels, inputLength, outChannels, kernelLength, width,
	)

	return inputBytes, weightBytes, biasBytes, encodeProjectionValuesAsDType(expected, storageDType)
}

func conv3DDTypeBytes(width int, storageDType dtype.DType) ([]byte, []byte, []byte, []byte) {
	batch, inChannels, outChannels := 1, 2, 2
	inputDepth, inputHeight, kernelDepth, kernelHeight, kernelWidth := 3, 3, 2, 2, 3
	inputWidth := width + kernelWidth - 1
	inputValues := projectionValues(batch*inChannels*inputDepth*inputHeight*inputWidth, 71, 64)
	weightValues := projectionValues(
		outChannels*inChannels*kernelDepth*kernelHeight*kernelWidth, 41, 128,
	)
	biasValues := projectionValues(outChannels, 19, 32)
	inputBytes := encodeProjectionValuesAsDType(inputValues, storageDType)
	weightBytes := encodeProjectionValuesAsDType(weightValues, storageDType)
	biasBytes := encodeProjectionValuesAsDType(biasValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	weightStored := decodeDTypeBytesToFloat32(weightBytes, storageDType)
	biasStored := decodeDTypeBytesToFloat32(biasBytes, storageDType)
	expected := conv3DExpectedValues(
		inputStored, weightStored, biasStored,
		batch, inChannels, inputDepth, inputHeight, inputWidth,
		outChannels, kernelDepth, kernelHeight, kernelWidth, 2, 2, width,
	)

	return inputBytes, weightBytes, biasBytes, encodeProjectionValuesAsDType(expected, storageDType)
}

func convTranspose2DDTypeBytes(width int, storageDType dtype.DType) ([]byte, []byte, []byte, []byte) {
	batch, inChannels, outChannels := 2, 2, 3
	inputHeight, kernelHeight, kernelWidth := 3, 2, 3
	outputHeight, outputWidth := inputHeight+kernelHeight-1, width+kernelWidth-1
	inputValues := projectionValues(batch*inChannels*inputHeight*width, 73, 64)
	weightValues := projectionValues(outChannels*inChannels*kernelHeight*kernelWidth, 43, 128)
	biasValues := projectionValues(outChannels, 23, 32)
	inputBytes := encodeProjectionValuesAsDType(inputValues, storageDType)
	weightBytes := encodeProjectionValuesAsDType(weightValues, storageDType)
	biasBytes := encodeProjectionValuesAsDType(biasValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	weightStored := decodeDTypeBytesToFloat32(weightBytes, storageDType)
	biasStored := decodeDTypeBytesToFloat32(biasBytes, storageDType)
	expected := convTranspose2DExpectedValues(
		inputStored, weightStored, biasStored,
		batch, inChannels, inputHeight, width,
		outChannels, kernelHeight, kernelWidth, outputHeight, outputWidth,
	)

	return inputBytes, weightBytes, biasBytes, encodeProjectionValuesAsDType(expected, storageDType)
}

func conv1DExpectedValues(
	input []float32,
	weight []float32,
	bias []float32,
	batch int,
	inChannels int,
	inputLength int,
	outChannels int,
	kernelLength int,
	outputLength int,
) []float32 {
	out := make([]float32, batch*outChannels*outputLength)

	for batchIndex := range batch {
		for outChannel := range outChannels {
			for outPosition := range outputLength {
				outIndex := (batchIndex*outChannels+outChannel)*outputLength + outPosition
				out[outIndex] = conv1DExpectedCell(
					input, weight, bias, batchIndex, outChannel, outPosition,
					inChannels, inputLength, kernelLength,
				)
			}
		}
	}

	return out
}

func conv1DExpectedCell(
	input []float32,
	weight []float32,
	bias []float32,
	batchIndex int,
	outChannel int,
	outPosition int,
	inChannels int,
	inputLength int,
	kernelLength int,
) float32 {
	accumulator := bias[outChannel]

	for inChannel := range inChannels {
		for kernelPosition := range kernelLength {
			inputIndex := (batchIndex*inChannels+inChannel)*inputLength +
				outPosition + kernelPosition
			weightIndex := (outChannel*inChannels+inChannel)*kernelLength + kernelPosition
			accumulator += input[inputIndex] * weight[weightIndex]
		}
	}

	return accumulator
}

func conv3DExpectedValues(
	input []float32,
	weight []float32,
	bias []float32,
	batch int,
	inChannels int,
	inputDepth int,
	inputHeight int,
	inputWidth int,
	outChannels int,
	kernelDepth int,
	kernelHeight int,
	kernelWidth int,
	outputDepth int,
	outputHeight int,
	outputWidth int,
) []float32 {
	out := make([]float32, batch*outChannels*outputDepth*outputHeight*outputWidth)

	for batchIndex := range batch {
		for outChannel := range outChannels {
			for outPlane := range outputDepth {
				for outRow := range outputHeight {
					for outCol := range outputWidth {
						outIndex := (((batchIndex*outChannels+outChannel)*outputDepth+outPlane)*
							outputHeight+outRow)*outputWidth + outCol
						out[outIndex] = conv3DExpectedCell(
							input, weight, bias, batchIndex, outChannel, outPlane, outRow, outCol,
							inChannels, inputDepth, inputHeight, inputWidth,
							kernelDepth, kernelHeight, kernelWidth,
						)
					}
				}
			}
		}
	}

	return out
}

func conv3DExpectedCell(
	input []float32,
	weight []float32,
	bias []float32,
	batchIndex int,
	outChannel int,
	outPlane int,
	outRow int,
	outCol int,
	inChannels int,
	inputDepth int,
	inputHeight int,
	inputWidth int,
	kernelDepth int,
	kernelHeight int,
	kernelWidth int,
) float32 {
	accumulator := bias[outChannel]

	for inChannel := range inChannels {
		for kernelPlane := range kernelDepth {
			for kernelRow := range kernelHeight {
				for kernelCol := range kernelWidth {
					inputIndex := (((batchIndex*inChannels+inChannel)*inputDepth+
						outPlane+kernelPlane)*inputHeight+outRow+kernelRow)*
						inputWidth + outCol + kernelCol
					weightIndex := (((outChannel*inChannels+inChannel)*kernelDepth+
						kernelPlane)*kernelHeight+kernelRow)*kernelWidth + kernelCol
					accumulator += input[inputIndex] * weight[weightIndex]
				}
			}
		}
	}

	return accumulator
}

func convTranspose2DExpectedValues(
	input []float32,
	weight []float32,
	bias []float32,
	batch int,
	inChannels int,
	inputHeight int,
	inputWidth int,
	outChannels int,
	kernelHeight int,
	kernelWidth int,
	outputHeight int,
	outputWidth int,
) []float32 {
	out := make([]float32, batch*outChannels*outputHeight*outputWidth)

	for batchIndex := range batch {
		for outChannel := range outChannels {
			for outRow := range outputHeight {
				for outCol := range outputWidth {
					outIndex := ((batchIndex*outChannels+outChannel)*outputHeight+outRow)*
						outputWidth + outCol
					out[outIndex] = convTranspose2DExpectedCell(
						input, weight, bias, batchIndex, outChannel, outRow, outCol,
						inChannels, inputHeight, inputWidth, outChannels,
						kernelHeight, kernelWidth,
					)
				}
			}
		}
	}

	return out
}

func convTranspose2DExpectedCell(
	input []float32,
	weight []float32,
	bias []float32,
	batchIndex int,
	outChannel int,
	outRow int,
	outCol int,
	inChannels int,
	inputHeight int,
	inputWidth int,
	outChannels int,
	kernelHeight int,
	kernelWidth int,
) float32 {
	accumulator := bias[outChannel]

	for inChannel := range inChannels {
		for kernelRow := range kernelHeight {
			if outRow < kernelRow {
				continue
			}

			inputRow := outRow - kernelRow
			if inputRow >= inputHeight {
				continue
			}

			for kernelCol := range kernelWidth {
				if outCol < kernelCol {
					continue
				}

				inputCol := outCol - kernelCol
				if inputCol >= inputWidth {
					continue
				}

				inputIndex := ((batchIndex*inChannels+inChannel)*inputHeight+inputRow)*
					inputWidth + inputCol
				weightIndex := ((inChannel*outChannels+outChannel)*kernelHeight+kernelRow)*
					kernelWidth + kernelCol
				accumulator += input[inputIndex] * weight[weightIndex]
			}
		}
	}

	return accumulator
}
