//go:build arm64

package kernels

func conv1DFloat32Native(
	config Conv1DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inLength,
	outChannels, kernelLength, outLength int,
) {
	if !conv1DConfigNEONEligible(config) {
		conv1DFloat32Scalar(
			config,
			inputView, weightView, biasView, outputView,
			batch, inChannels, inLength, outChannels, kernelLength, outLength,
		)

		return
	}

	inWStride := inLength
	inCStride := inLength
	weightWStride := kernelLength
	weightCStride := kernelLength

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inLength

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelLength
			outputChannelOffset := (batchIndex*outChannels + outChIndex) * outLength
			outputRow := outputView[outputChannelOffset : outputChannelOffset+outLength]
			blockCols := len(outputRow) &^ 3

			if blockCols > 0 {
				conv2dStride1RowNEONAsm(
					&outputRow[0],
					&inputView[inputBatchOffset],
					&weightView[weightChannelOffset],
					biasView[outChIndex],
					blockCols,
					inChannels, 1, kernelLength,
					inWStride, inCStride,
					weightWStride, weightCStride,
					0, 0,
				)
			}

			for outIndex := blockCols; outIndex < outLength; outIndex++ {
				outputRow[outIndex] = conv1DPixelScalar(
					config,
					inputView, weightView,
					inputBatchOffset, weightChannelOffset,
					inChannels, inLength, kernelLength,
					outIndex,
					biasView[outChIndex],
				)
			}
		}
	}
}

func conv1DConfigNEONEligible(config Conv1DConfig) bool {
	return config.Stride == 1 &&
		config.Padding == 0 &&
		config.Dilation == 1
}
