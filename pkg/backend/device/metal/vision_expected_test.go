package metal

import "github.com/theapemachine/caramba/pkg/dtype"

func conv2DDTypeBytes(
	outputWidth int,
	storageDType dtype.DType,
) ([]byte, []byte, []byte, []byte) {
	batch, inChannels, outChannels := 2, 2, 3
	inputHeight, kernelHeight, kernelWidth := 4, 2, 3
	outputHeight := inputHeight - kernelHeight + 1
	inputWidth := outputWidth + kernelWidth - 1
	inputValues := projectionValues(batch*inChannels*inputHeight*inputWidth, 59, 64)
	weightValues := projectionValues(outChannels*inChannels*kernelHeight*kernelWidth, 31, 128)
	biasValues := projectionValues(outChannels, 13, 32)
	inputBytes := encodeProjectionValuesAsDType(inputValues, storageDType)
	weightBytes := encodeProjectionValuesAsDType(weightValues, storageDType)
	biasBytes := encodeProjectionValuesAsDType(biasValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	weightStored := decodeDTypeBytesToFloat32(weightBytes, storageDType)
	biasStored := decodeDTypeBytesToFloat32(biasBytes, storageDType)
	expected := conv2DExpectedValues(
		inputStored, weightStored, biasStored,
		batch, inChannels, inputHeight, inputWidth,
		outChannels, kernelHeight, kernelWidth, outputHeight, outputWidth,
	)

	return inputBytes, weightBytes, biasBytes, encodeProjectionValuesAsDType(expected, storageDType)
}

func pool2DDTypeBytes(outputWidth int, storageDType dtype.DType) ([]byte, []byte, []byte) {
	batch, channels, inputHeight := 2, 3, 4
	inputWidth := outputWidth * 2
	inputValues := projectionValues(batch*channels*inputHeight*inputWidth, 53, 64)
	inputBytes := encodeProjectionValuesAsDType(inputValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	maxExpected := pool2DExpectedValues(inputStored, batch, channels, inputHeight, inputWidth, true)
	avgExpected := pool2DExpectedValues(inputStored, batch, channels, inputHeight, inputWidth, false)

	return inputBytes,
		encodeProjectionValuesAsDType(maxExpected, storageDType),
		encodeProjectionValuesAsDType(avgExpected, storageDType)
}

func adaptivePool2DDTypeBytes(outputWidth int, storageDType dtype.DType) ([]byte, []byte, []byte) {
	batch, channels, inputHeight := 2, 2, 5
	inputWidth := outputWidth + 3
	outHeight := 3
	inputValues := projectionValues(batch*channels*inputHeight*inputWidth, 61, 64)
	inputBytes := encodeProjectionValuesAsDType(inputValues, storageDType)
	inputStored := decodeDTypeBytesToFloat32(inputBytes, storageDType)
	avgExpected := adaptivePool2DExpectedValues(
		inputStored, batch, channels, inputHeight, inputWidth, outHeight, outputWidth, false,
	)
	maxExpected := adaptivePool2DExpectedValues(
		inputStored, batch, channels, inputHeight, inputWidth, outHeight, outputWidth, true,
	)

	return inputBytes,
		encodeProjectionValuesAsDType(avgExpected, storageDType),
		encodeProjectionValuesAsDType(maxExpected, storageDType)
}

func conv2DExpectedValues(
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
					out[outIndex] = conv2DExpectedCell(
						input, weight, bias, batchIndex, outChannel, outRow, outCol,
						inChannels, inputHeight, inputWidth, kernelHeight, kernelWidth,
					)
				}
			}
		}
	}

	return out
}

func conv2DExpectedCell(
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
	kernelHeight int,
	kernelWidth int,
) float32 {
	accumulator := bias[outChannel]

	for inChannel := range inChannels {
		for kernelRow := range kernelHeight {
			for kernelCol := range kernelWidth {
				inputIndex := ((batchIndex*inChannels+inChannel)*inputHeight+outRow+kernelRow)*
					inputWidth + outCol + kernelCol
				weightIndex := ((outChannel*inChannels+inChannel)*kernelHeight+kernelRow)*
					kernelWidth + kernelCol
				accumulator += input[inputIndex] * weight[weightIndex]
			}
		}
	}

	return accumulator
}

func pool2DExpectedValues(
	input []float32,
	batch int,
	channels int,
	inputHeight int,
	inputWidth int,
	useMax bool,
) []float32 {
	outputHeight := inputHeight / 2
	outputWidth := inputWidth / 2
	out := make([]float32, batch*channels*outputHeight*outputWidth)

	for batchIndex := range batch {
		for channel := range channels {
			for outRow := range outputHeight {
				for outCol := range outputWidth {
					outIndex := ((batchIndex*channels+channel)*outputHeight+outRow)*
						outputWidth + outCol
					out[outIndex] = pool2DExpectedCell(
						input, batchIndex, channel, channels, inputHeight, inputWidth,
						outRow, outCol, useMax,
					)
				}
			}
		}
	}

	return out
}

func pool2DExpectedCell(
	input []float32,
	batchIndex int,
	channel int,
	channels int,
	inputHeight int,
	inputWidth int,
	outRow int,
	outCol int,
	useMax bool,
) float32 {
	value := float32(0)
	if useMax {
		value = -1.0e30
	}

	for kernelRow := range 2 {
		for kernelCol := range 2 {
			inputIndex := ((batchIndex*channels+channel)*inputHeight+outRow*2+kernelRow)*
				inputWidth + outCol*2 + kernelCol
			candidate := input[inputIndex]

			if useMax && candidate > value {
				value = candidate
				continue
			}

			if !useMax {
				value += candidate
			}
		}
	}

	if useMax {
		return value
	}

	return value * 0.25
}

func adaptivePool2DExpectedValues(
	input []float32,
	batch int,
	channels int,
	inputHeight int,
	inputWidth int,
	outputHeight int,
	outputWidth int,
	useMax bool,
) []float32 {
	out := make([]float32, batch*channels*outputHeight*outputWidth)

	for batchIndex := range batch {
		for channel := range channels {
			for outRow := range outputHeight {
				for outCol := range outputWidth {
					outIndex := ((batchIndex*channels+channel)*outputHeight+outRow)*
						outputWidth + outCol
					out[outIndex] = adaptivePool2DExpectedCell(
						input, batchIndex, channel, channels, inputHeight, inputWidth,
						outputHeight, outputWidth, outRow, outCol, useMax,
					)
				}
			}
		}
	}

	return out
}

func adaptivePool2DExpectedCell(
	input []float32,
	batchIndex int,
	channel int,
	channels int,
	inputHeight int,
	inputWidth int,
	outputHeight int,
	outputWidth int,
	outRow int,
	outCol int,
	useMax bool,
) float32 {
	startRow := outRow * inputHeight / outputHeight
	endRow := (outRow + 1) * inputHeight / outputHeight
	startCol := outCol * inputWidth / outputWidth
	endCol := (outCol + 1) * inputWidth / outputWidth
	value := float32(0)
	elements := 0

	if useMax {
		value = -1.0e30
	}

	for inputRow := startRow; inputRow < endRow; inputRow++ {
		for inputCol := startCol; inputCol < endCol; inputCol++ {
			inputIndex := ((batchIndex*channels+channel)*inputHeight+inputRow)*inputWidth + inputCol
			candidate := input[inputIndex]
			elements++

			if useMax && candidate > value {
				value = candidate
				continue
			}

			if !useMax {
				value += candidate
			}
		}
	}

	if useMax || elements == 0 {
		return value
	}

	return value / float32(elements)
}
