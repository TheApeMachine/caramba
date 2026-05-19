package neon

func conv1DPixelScalar(
	config Conv1DConfig,
	inputView, weightView []float32,
	inputBatchOffset, weightChannelOffset int,
	inChannels, inLength, kernelLength, outIndex int,
	biasValue float32,
) float32 {
	sum := biasValue

	for inChIndex := range inChannels {
		for kernelIndex := range kernelLength {
			inPos := outIndex*config.Stride + kernelIndex*config.Dilation - config.Padding

			if inPos < 0 || inPos >= inLength {
				continue
			}

			sum += inputView[inputBatchOffset+inChIndex*inLength+inPos] *
				weightView[weightChannelOffset+inChIndex*kernelLength+kernelIndex]
		}
	}

	return sum
}

func conv1DFloat32Scalar(
	config Conv1DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
) {
	for batchIndex := range batch {
		for outChIndex := range outChannels {
			for outIndex := range outLength {
				outputView[(batchIndex*outChannels+outChIndex)*outLength+outIndex] =
					conv1DPixelScalar(
						config,
						inputView, weightView,
						batchIndex*inChannels*inLength,
						outChIndex*inChannels*kernelLength,
						inChannels, inLength, kernelLength, outIndex,
						biasView[outChIndex],
					)
			}
		}
	}
}
