package convolution

import "unsafe"

func Conv2DFloat32Scalar(
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

	for batchIndex := range batch {
		for outChIndex := range outChannels {
			for outRow := range outHeight {
				for outCol := range outWidth {
					outIndex := ((batchIndex*outChannels+outChIndex)*outHeight+outRow)*outWidth + outCol
					outputView[outIndex] = Conv2DPixelScalar(
						config,
						inputView, weightView,
						batchIndex*inChannels*inHeight*inWidth,
						outChIndex*inChannels*kernelHeight*kernelWidth,
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

func Conv2DPixelScalar(
	config Conv2DConfig,
	inputView, weightView []float32,
	inputBatchOffset, weightChannelOffset int,
	inChannels, inHeight, inWidth,
	kernelHeight, kernelWidth,
	outRow, outCol int,
	biasValue float32,
) float32 {
	sum := biasValue

	for inChIndex := range inChannels {
		for kRow := range kernelHeight {
			inRow := outRow*config.StrideH + kRow*config.DilationH - config.PaddingH

			if inRow < 0 || inRow >= inHeight {
				continue
			}

			for kCol := range kernelWidth {
				inCol := outCol*config.StrideW + kCol*config.DilationW - config.PaddingW

				if inCol < 0 || inCol >= inWidth {
					continue
				}

				inputIndex := inputBatchOffset + (inChIndex*inHeight+inRow)*inWidth + inCol
				weightIndex := weightChannelOffset + (inChIndex*kernelHeight+kRow)*kernelWidth + kCol

				sum += inputView[inputIndex] * weightView[weightIndex]
			}
		}
	}

	return sum
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
