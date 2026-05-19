package convolution

import "unsafe"

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

func Conv1DFloat32Scalar(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
) {
	inputView := float32View(input, batch*inChannels*inLength)
	weightView := float32View(weight, outChannels*inChannels*kernelLength)
	biasView := float32View(bias, outChannels)
	outputView := float32View(output, batch*outChannels*outLength)

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

func conv1DConfigNEONEligible(config Conv1DConfig) bool {
	return config.Stride == 1 &&
		config.Padding == 0 &&
		config.Dilation == 1
}
