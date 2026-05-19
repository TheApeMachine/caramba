//go:build !arm64

package kernels

func convTranspose2DConfigNEONEligible(config Conv2DConfig) bool {
	return config.StrideH == 1 &&
		config.StrideW == 1 &&
		config.PaddingH == 0 &&
		config.PaddingW == 0 &&
		config.DilationH == 1 &&
		config.DilationW == 1
}

func convTranspose2DFloat32Scalar(
	config Conv2DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	convTranspose2DFloat32InitBias(
		outputView, biasView,
		batch, outChannels, outHeight, outWidth,
	)

	for batchIndex := range batch {
		convTranspose2DBatchScalar(
			config, inputView, weightView, outputView,
			batchIndex, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth, outHeight, outWidth,
		)
	}
}

func convTranspose2DBatchScalar(
	config Conv2DConfig,
	inputView, weightView, outputView []float32,
	batchIndex, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	inputBatchOffset := batchIndex * inChannels * inHeight * inWidth
	outputBatchOffset := batchIndex * outChannels * outHeight * outWidth

	for inChIndex := range inChannels {
		for outChIndex := range outChannels {
			convTranspose2DChannelScalar(
				config, inputView, weightView, outputView, inputBatchOffset,
				outputBatchOffset, inChIndex, outChIndex, inHeight, inWidth,
				outChannels, kernelHeight, kernelWidth, outHeight, outWidth,
			)
		}
	}
}

func convTranspose2DChannelScalar(
	config Conv2DConfig,
	inputView, weightView, outputView []float32,
	inputBatchOffset, outputBatchOffset, inChIndex, outChIndex,
	inHeight, inWidth, outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	inputChannelOffset := inputBatchOffset + inChIndex*inHeight*inWidth
	weightChannelOffset := (inChIndex*outChannels + outChIndex) * kernelHeight * kernelWidth
	outputChannelOffset := outputBatchOffset + outChIndex*outHeight*outWidth

	for inRow := range inHeight {
		for inCol := range inWidth {
			inputValue := inputView[inputChannelOffset+inRow*inWidth+inCol]
			convTranspose2DScatterScalar(
				config, weightView, outputView, inputValue,
				weightChannelOffset, outputChannelOffset, inRow, inCol,
				kernelHeight, kernelWidth, outHeight, outWidth,
			)
		}
	}
}

func convTranspose2DScatterScalar(
	config Conv2DConfig,
	weightView, outputView []float32,
	inputValue float32,
	weightChannelOffset, outputChannelOffset, inRow, inCol,
	kernelHeight, kernelWidth, outHeight, outWidth int,
) {
	for kRow := range kernelHeight {
		outRow := inRow*config.StrideH + kRow*config.DilationH - config.PaddingH
		if outRow < 0 || outRow >= outHeight {
			continue
		}

		convTranspose2DScatterRowScalar(
			config, weightView, outputView, inputValue, weightChannelOffset,
			outputChannelOffset, inCol, kRow, kernelWidth, outRow, outWidth,
		)
	}
}

func convTranspose2DScatterRowScalar(
	config Conv2DConfig,
	weightView, outputView []float32,
	inputValue float32,
	weightChannelOffset, outputChannelOffset, inCol, kRow, kernelWidth, outRow, outWidth int,
) {
	outColBase := inCol*config.StrideW - config.PaddingW

	for kCol := range kernelWidth {
		outCol := outColBase + kCol*config.DilationW
		if outCol < 0 || outCol >= outWidth {
			continue
		}

		weightIndex := weightChannelOffset + kRow*kernelWidth + kCol
		outIndex := outputChannelOffset + outRow*outWidth + outCol
		outputView[outIndex] += inputValue * weightView[weightIndex]
	}
}

func convTranspose2DFloat32InitBias(
	outputView, biasView []float32,
	batch, outChannels, outHeight, outWidth int,
) {
	for batchIndex := range batch {
		for outChIndex := range outChannels {
			channelOffset := (batchIndex*outChannels + outChIndex) * outHeight * outWidth
			channel := outputView[channelOffset : channelOffset+outHeight*outWidth]

			for index := range channel {
				channel[index] = biasView[outChIndex]
			}
		}
	}
}

func convTranspose2DPixelScalar(
	config Conv2DConfig,
	inputView, weightView []float32,
	inputChannelOffset, weightChannelOffset,
	inHeight, inWidth, kernelHeight, kernelWidth,
	outRow, outCol int,
) float32 {
	var sum float32

	for kernelRow := range kernelHeight {
		inputRowNumerator := outRow + config.PaddingH - kernelRow*config.DilationH
		if inputRowNumerator%config.StrideH != 0 {
			continue
		}

		inputRow := inputRowNumerator / config.StrideH
		if inputRow < 0 || inputRow >= inHeight {
			continue
		}

		sum += convTranspose2DPixelRowScalar(
			config, inputView, weightView, inputChannelOffset, weightChannelOffset,
			inputRow, outCol, inWidth, kernelRow, kernelWidth,
		)
	}

	return sum
}

func convTranspose2DPixelRowScalar(
	config Conv2DConfig,
	inputView, weightView []float32,
	inputChannelOffset, weightChannelOffset,
	inputRow, outCol, inWidth, kernelRow, kernelWidth int,
) float32 {
	var sum float32

	for kernelCol := range kernelWidth {
		inputColNumerator := outCol + config.PaddingW - kernelCol*config.DilationW
		if inputColNumerator%config.StrideW != 0 {
			continue
		}

		inputCol := inputColNumerator / config.StrideW
		if inputCol < 0 || inputCol >= inWidth {
			continue
		}

		inputValue := inputView[inputChannelOffset+inputRow*inWidth+inputCol]
		weightValue := weightView[weightChannelOffset+kernelRow*kernelWidth+kernelCol]
		sum += inputValue * weightValue
	}

	return sum
}
