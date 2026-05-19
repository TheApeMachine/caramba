//go:build arm64

package convolution

import "unsafe"

func Conv2DFloat32Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if conv2DConfigNEONEligible(config) {
		Conv2DFloat32Stride1RowNative(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	Conv2DFloat32GeneralNative(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func Conv2DFloat32Stride1RowNative(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	inputView := float32View(input, batch*inChannels*inHeight*inWidth)
	weightView := float32View(weight, outChannels*inChannels*kernelHeight*kernelWidth)
	biasView := float32View(bias, outChannels)
	outputView := float32View(output, batch*outChannels*outHeight*outWidth)

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
				blockCols := len(outputRow) &^ 3

				if blockCols > 0 {
					Conv2dStride1RowNEONAsm(
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
					outputRow[outCol] = Conv2DPixelScalar(
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

func Conv2DFloat32GeneralNative(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	inputView := float32View(input, batch*inChannels*inHeight*inWidth)
	weightView := float32View(weight, outChannels*inChannels*kernelHeight*kernelWidth)
	biasView := float32View(bias, outChannels)
	outputView := float32View(output, batch*outChannels*outHeight*outWidth)

	patchLength := inChannels * kernelHeight * kernelWidth
	patchScratch := BorrowFloat32Buffer(patchLength)
	defer ReleaseFloat32Buffer(patchScratch)

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inHeight * inWidth

		for outChIndex := range outChannels {
			weightChannelOffset := outChIndex * inChannels * kernelHeight * kernelWidth

			for outRow := range outHeight {
				for outCol := range outWidth {
					Conv2DPatchGather(
						config,
						inputView, inputBatchOffset,
						patchScratch,
						inChannels, inHeight, inWidth,
						kernelHeight, kernelWidth,
						outRow, outCol,
					)

					dotValue := Conv2dPatchDotNEONAsm(
						&weightView[weightChannelOffset],
						&patchScratch[0],
						patchLength,
					)

					outputView[((batchIndex*outChannels+outChIndex)*outHeight+outRow)*outWidth+outCol] =
						biasView[outChIndex] + dotValue
				}
			}
		}
	}
}

func Conv1DFloat32Native(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength,
	outChannels, kernelLength, outLength int,
) {
	if !conv1DConfigNEONEligible(config) {
		Conv1DFloat32Scalar(
			config,
			input, weight, bias, output,
			batch, inChannels, inLength, outChannels, kernelLength, outLength,
		)

		return
	}

	inputView := float32View(input, batch*inChannels*inLength)
	weightView := float32View(weight, outChannels*inChannels*kernelLength)
	biasView := float32View(bias, outChannels)
	outputView := float32View(output, batch*outChannels*outLength)

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
				Conv2dStride1RowNEONAsm(
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

func Conv3DFloat32Native(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	inputView := float32View(input, batch*inChannels*inD*inH*inW)
	weightView := float32View(weight, outChannels*inChannels*kD*kH*kW)
	biasView := float32View(bias, outChannels)
	outputView := float32View(output, batch*outChannels*outD*outH*outW)

	patchLength := inChannels * kD * kH * kW
	patchScratch := BorrowFloat32Buffer(patchLength)
	defer ReleaseFloat32Buffer(patchScratch)

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inD * inH * inW

		for outChIndex := range outChannels {
			weightOffset := outChIndex * inChannels * kD * kH * kW

			for outDIndex := range outD {
				for outHIndex := range outH {
					for outWIndex := range outW {
						conv3DPatchGather(
							config,
							inputView, inputBatchOffset,
							patchScratch,
							inChannels, inD, inH, inW,
							kD, kH, kW,
							outDIndex, outHIndex, outWIndex,
						)

						dotValue := Conv3dPatchDotNEONAsm(
							&weightView[weightOffset],
							&patchScratch[0],
							patchLength,
						)

						outputView[(((batchIndex*outChannels+outChIndex)*outD+outDIndex)*outH+outHIndex)*outW+outWIndex] =
							biasView[outChIndex] + dotValue
					}
				}
			}
		}
	}
}

func ConvTranspose2DFloat32Native(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if !ConvTranspose2DConfigNEONEligible(config) {
		ConvTranspose2DFloat32Scalar(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	ConvTranspose2DFloat32EligibleNative(
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func ConvTranspose2DFloat32EligibleNative(
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	inputView := float32View(input, batch*inChannels*inHeight*inWidth)
	weightView := float32View(weight, inChannels*outChannels*kernelHeight*kernelWidth)
	biasView := float32View(bias, outChannels)
	outputView := float32View(output, batch*outChannels*outHeight*outWidth)

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
