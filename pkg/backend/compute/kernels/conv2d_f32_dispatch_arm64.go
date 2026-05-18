//go:build arm64

package kernels

func conv2DFloat32Native(
	config Conv2DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if !conv2DConfigNEONEligible(config) {
		conv2DFloat32Scalar(
			config,
			inputView, weightView, biasView, outputView,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	inHStride := inWidth
	inCStride := inHeight * inWidth
	weightHStride := kernelWidth
	weightCStride := kernelHeight * kernelWidth

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inHeight * inWidth

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelHeight * kernelWidth
			outputChannelOffset := (batchIndex*outChannels + outChIndex) * outHeight * outWidth

			for outRow := range outHeight {
				outputRow := outputView[outputChannelOffset+outRow*outWidth : outputChannelOffset+(outRow+1)*outWidth]
				blockCols := len(outputRow) & ^3

				if blockCols > 0 {
					conv2dStride1RowNEONAsm(
						&outputRow[0],
						&inputView[inputBatchOffset],
						&weightView[weightChannelOffset],
						biasView[outChIndex],
						blockCols,
						inChannels, kernelHeight, kernelWidth,
						inHStride, inCStride,
						weightHStride, weightCStride,
						outRow, 0,
					)
				}

				for outCol := blockCols; outCol < outWidth; outCol++ {
					outputRow[outCol] = conv2DPixelScalar(
						config,
						inputView, weightView,
						inputBatchOffset, weightChannelOffset,
						inChannels, inHeight, inWidth,
						kernelHeight, kernelWidth,
						outRow, outCol,
						biasView[outChIndex],
					)
				}
			}
		}
	}
}

func conv2DConfigNEONEligible(config Conv2DConfig) bool {
	return config.StrideH == 1 &&
		config.StrideW == 1 &&
		config.PaddingH == 0 &&
		config.PaddingW == 0 &&
		config.DilationH == 1 &&
		config.DilationW == 1
}
