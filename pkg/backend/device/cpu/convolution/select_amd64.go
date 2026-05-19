//go:build amd64

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
		Conv2DFloat32Scalar(
			config,
			input, weight, bias, output,
			batch, inChannels, inHeight, inWidth,
			outChannels, kernelHeight, kernelWidth,
			outHeight, outWidth,
		)

		return
	}

	conv2DFloat32GeneralAVX512(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}

func conv2DFloat32GeneralAVX512(
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

					dotValue := convPatchDotF32Native(
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
	Conv1DFloat32Scalar(
		config,
		input, weight, bias, output,
		batch, inChannels, inLength, outChannels, kernelLength, outLength,
	)
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

						dotValue := convPatchDotF32Native(
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
	ConvTranspose2DFloat32Scalar(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
	)
}
