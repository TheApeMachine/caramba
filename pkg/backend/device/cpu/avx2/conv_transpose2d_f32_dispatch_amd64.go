//go:build amd64

package avx2

import "golang.org/x/sys/cpu"

func ConvTranspose2DFloat32(
	config Conv2DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if !cpu.X86.HasAVX2 || !convTranspose2DConfigAVX2Eligible(config) {
		ConvTranspose2DFloat32Scalar(
			config,
			inputView, weightView, biasView, outputView,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	convTranspose2DFloat32EligibleAVX2(
		inputView, weightView, biasView, outputView,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func convTranspose2DFloat32EligibleAVX2(
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	convTranspose2DFloat32InitBias(
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
							outputRow[outCol] += convTranspose2DPixelScalar(
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
						outputRow[outCol] += convTranspose2DPixelScalar(
							DefaultConv2DConfig(),
							inputView, weightView,
							inputChannelOffset, weightChannelOffset,
							inHeight, inWidth,
							kernelHeight, kernelWidth,
							outRow, outCol,
						)
					}

					blockCols := (outWidth - scalarPrefix) &^ 7
					inputBlockCols := (inWidth - scalarPrefix) &^ 7

					if inputBlockCols < blockCols {
						blockCols = inputBlockCols
					}

					if blockCols > 0 {
						convTranspose2dStride1RowAVX2(
							outputRow[scalarPrefix:],
							inputView[inputChannelOffset:],
							weightView[weightChannelOffset:],
							blockCols,
							kernelHeight, kernelWidth, inHeight, inWidth,
							outRow, scalarPrefix,
						)
					}

					for outCol := scalarPrefix + blockCols; outCol < outWidth; outCol++ {
						outputRow[outCol] += convTranspose2DPixelScalar(
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
