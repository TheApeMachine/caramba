//go:build !arm64

package kernels

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
		inputBatchOffset := batchIndex * inChannels * inHeight * inWidth
		outputBatchOffset := batchIndex * outChannels * outHeight * outWidth

		for inChIndex := range inChannels {
			for inRow := range inHeight {
				for inCol := range inWidth {
					inputValue := inputView[inputBatchOffset+inChIndex*inHeight*inWidth+inRow*inWidth+inCol]

					for outChIndex := range outChannels {
						for kRow := range kernelHeight {
							outRow := inRow*config.StrideH + kRow*config.DilationH - config.PaddingH
							outColBase := inCol*config.StrideW - config.PaddingW

							if outRow < 0 || outRow >= outHeight {
								continue
							}

							for kCol := range kernelWidth {
								outCol := outColBase + kCol*config.DilationW

								if outCol < 0 || outCol >= outWidth {
									continue
								}

								weightIndex := ((inChIndex*outChannels+outChIndex)*kernelHeight+kRow)*kernelWidth + kCol
								outIndex := outputBatchOffset + outChIndex*outHeight*outWidth + outRow*outWidth + outCol
								outputView[outIndex] += inputValue * weightView[weightIndex]
							}
						}
					}
				}
			}
		}
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

func convTranspose2DConfigNEONEligible(config Conv2DConfig) bool {
	return false
}
