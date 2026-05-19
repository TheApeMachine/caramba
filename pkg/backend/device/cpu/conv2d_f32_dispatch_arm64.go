//go:build arm64

package cpu

func Conv2DFloat32Native(
	config Conv2DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
	if conv2DConfigNEONEligible(config) {
		Conv2DFloat32Stride1RowNative(
			config,
			inputView, weightView, biasView, outputView,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	Conv2DFloat32GeneralNative(
		config,
		inputView, weightView, biasView, outputView,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func Conv2DFloat32Stride1RowNative(
	config Conv2DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
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
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
) {
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

func Conv2DPatchGather(
	config Conv2DConfig,
	inputView []float32,
	inputBatchOffset int,
	patchScratch []float32,
	inChannels, inHeight, inWidth, kernelHeight, kernelWidth, outRow, outCol int,
) {
	patchIndex := 0

	for inChIndex := range inChannels {
		for kRow := range kernelHeight {
			inRow := outRow*config.StrideH + kRow*config.DilationH - config.PaddingH

			for kCol := range kernelWidth {
				inCol := outCol*config.StrideW + kCol*config.DilationW - config.PaddingW
				value := float32(0)

				if inRow >= 0 && inRow < inHeight && inCol >= 0 && inCol < inWidth {
					inputIndex := inputBatchOffset + (inChIndex*inHeight+inRow)*inWidth + inCol
					value = inputView[inputIndex]
				}

				patchScratch[patchIndex] = value
				patchIndex++
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
