package convolution

import "unsafe"

func conv3DPatchGather(
	config Conv3DConfig,
	inputView []float32,
	inputBatchOffset int,
	patchScratch []float32,
	inChannels, inD, inH, inW, kD, kH, kW, outDIndex, outHIndex, outWIndex int,
) {
	patchIndex := 0

	for inChIndex := range inChannels {
		for kDIndex := range kD {
			inDPos := outDIndex*config.StrideD + kDIndex*config.DilationD - config.PaddingD

			for kHIndex := range kH {
				inHPos := outHIndex*config.StrideH + kHIndex*config.DilationH - config.PaddingH

				for kWIndex := range kW {
					inWPos := outWIndex*config.StrideW + kWIndex*config.DilationW - config.PaddingW
					value := float32(0)

					if inDPos >= 0 && inDPos < inD &&
						inHPos >= 0 && inHPos < inH &&
						inWPos >= 0 && inWPos < inW {
						inputIndex := inputBatchOffset +
							((inChIndex*inD+inDPos)*inH+inHPos)*inW + inWPos
						value = inputView[inputIndex]
					}

					patchScratch[patchIndex] = value
					patchIndex++
				}
			}
		}
	}
}

func Conv3DFloat32Scalar(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	inputView := float32View(input, batch*inChannels*inD*inH*inW)
	weightView := float32View(weight, outChannels*inChannels*kD*kH*kW)
	biasView := float32View(bias, outChannels)
	outputView := float32View(output, batch*outChannels*outD*outH*outW)

	for batchIndex := range batch {
		for outChIndex := range outChannels {
			for outDIndex := range outD {
				for outHIndex := range outH {
					for outWIndex := range outW {
						outputView[(((batchIndex*outChannels+outChIndex)*outD+outDIndex)*outH+outHIndex)*outW+outWIndex] =
							conv3DPixelScalar(
								config,
								inputView, weightView,
								batchIndex, outChIndex,
								inChannels, inD, inH, inW,
								kD, kH, kW,
								outDIndex, outHIndex, outWIndex,
								biasView[outChIndex],
							)
					}
				}
			}
		}
	}
}

func conv3DPixelScalar(
	config Conv3DConfig,
	inputView, weightView []float32,
	batchIndex, outChIndex, inChannels, inD, inH, inW, kD, kH, kW, outDIndex, outHIndex, outWIndex int,
	biasValue float32,
) float32 {
	sum := biasValue

	for inChIndex := range inChannels {
		for kDIndex := range kD {
			for kHIndex := range kH {
				for kWIndex := range kW {
					inDPos := outDIndex*config.StrideD + kDIndex*config.DilationD - config.PaddingD
					inHPos := outHIndex*config.StrideH + kHIndex*config.DilationH - config.PaddingH
					inWPos := outWIndex*config.StrideW + kWIndex*config.DilationW - config.PaddingW

					if inDPos < 0 || inDPos >= inD ||
						inHPos < 0 || inHPos >= inH ||
						inWPos < 0 || inWPos >= inW {
						continue
					}

					inputIndex := (((batchIndex*inChannels+inChIndex)*inD+inDPos)*inH+inHPos)*inW + inWPos
					weightIndex := (((outChIndex*inChannels+inChIndex)*kD+kDIndex)*kH+kHIndex)*kW + kWIndex
					sum += inputView[inputIndex] * weightView[weightIndex]
				}
			}
		}
	}

	return sum
}
