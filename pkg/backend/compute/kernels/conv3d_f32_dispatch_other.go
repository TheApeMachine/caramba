//go:build !arm64

package kernels

func conv3DFloat32Native(
	config Conv3DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	conv3DFloat32Scalar(
		config,
		inputView, weightView, biasView, outputView,
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW,
	)
}

func conv3DFloat32Scalar(
	config Conv3DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
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
