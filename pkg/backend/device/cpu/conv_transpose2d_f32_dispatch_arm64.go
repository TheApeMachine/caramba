//go:build arm64

package cpu

func ConvTranspose2DFloat32Native(
	config Conv2DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if !ConvTranspose2DConfigNEONEligible(config) {
		ConvTranspose2DFloat32Scalar(
			config,
			inputView, weightView, biasView, outputView,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	ConvTranspose2DFloat32EligibleNative(
		inputView, weightView, biasView, outputView,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func ConvTranspose2DFloat32EligibleNative(
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	ConvTranspose2DFloat32InitBias(
		outputView, biasView,
		batch, outChannels, outHeight, outWidth,
	)

	inCStride := inHeight * inWidth
	weightOutChStride := kernelHeight * kernelWidth
	weightInChStride := outChannels * weightOutChStride

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inCStride
		outputBatchOffset := batchIndex * outChannels * outHeight * outWidth

		for outChIndex := range outChannels {
			outputChannelOffset := outputBatchOffset + outChIndex*outHeight*outWidth

			for inChIndex := range inChannels {
				inputChannelOffset := inputBatchOffset + inChIndex*inCStride
				weightChannelOffset := inChIndex*weightInChStride + outChIndex*weightOutChStride

				for outRow := range outHeight {
					outputRow := outputView[outputChannelOffset+outRow*outWidth : outputChannelOffset+(outRow+1)*outWidth]
					scalarPrefix := kernelWidth - 1

					if scalarPrefix > outWidth {
						scalarPrefix = outWidth
					}

					if outRow < kernelHeight-1 {
						for outCol := range outWidth {
							outputRow[outCol] += ConvTranspose2DPixelScalar(
								DefaultConv2DConfig(),
								inputView, weightView,
								inputChannelOffset, weightChannelOffset,
								inHeight, inWidth,
								kernelHeight, kernelWidth,
								outRow, outCol,
							)
						}

						continue
					}

					for outCol := 0; outCol < scalarPrefix; outCol++ {
						outputRow[outCol] += ConvTranspose2DPixelScalar(
							DefaultConv2DConfig(),
							inputView, weightView,
							inputChannelOffset, weightChannelOffset,
							inHeight, inWidth,
							kernelHeight, kernelWidth,
							outRow, outCol,
						)
					}

					blockCols := (outWidth - scalarPrefix) &^ 3
					inputBlockCols := (inWidth - scalarPrefix) &^ 3

					if inputBlockCols < blockCols {
						blockCols = inputBlockCols
					}

					if blockCols > 0 {
						ConvTranspose2dStride1RowNEON(
							outputRow[scalarPrefix:],
							inputView[inputChannelOffset:],
							weightView[weightChannelOffset:],
							blockCols,
							kernelHeight, kernelWidth, inHeight, inWidth,
							outRow, scalarPrefix,
						)
					}

					for outCol := scalarPrefix + blockCols; outCol < outWidth; outCol++ {
						outputRow[outCol] += ConvTranspose2DPixelScalar(
							DefaultConv2DConfig(),
							inputView, weightView,
							inputChannelOffset, weightChannelOffset,
							inHeight, inWidth,
							kernelHeight, kernelWidth,
							outRow, outCol,
						)
					}
				}
			}
		}
	}
}
