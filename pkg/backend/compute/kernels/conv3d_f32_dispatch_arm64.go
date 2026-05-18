//go:build arm64

package kernels

func conv3DFloat32Native(
	config Conv3DConfig,
	inputView, weightView, biasView, outputView []float32,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
) {
	if !conv3DConfigNEONEligible(config) {
		conv3DFloat32Scalar(
			config,
			inputView, weightView, biasView, outputView,
			batch, inChannels, inD, inH, inW,
			outChannels, kD, kH, kW, outD, outH, outW,
		)

		return
	}

	inHStride := inW
	inDStride := inH * inW
	inCStride := inD * inH * inW
	weightHStride := kW
	weightDStride := kH * kW
	weightCStride := kD * kH * kW
	rowScratch := borrowFloat32Buffer(outW)
	defer releaseFloat32Buffer(rowScratch)

	for batchIndex := range batch {
		inputBatchOffset := batchIndex * inChannels * inCStride

		for outChIndex := range outChannels {
			weightOutOffset := outChIndex * inChannels * weightCStride
			outputBatchOffset := batchIndex * outChannels * outD * outH * outW
			outputChannelOffset := outputBatchOffset + outChIndex*outD*outH*outW

			for outDIndex := range outD {
				for outHIndex := range outH {
					outputRowOffset := outputChannelOffset + (outDIndex*outH+outHIndex)*outW
					outputRow := outputView[outputRowOffset : outputRowOffset+outW]
					blockCols := outW &^ 3

					for outColIndex := range outputRow {
						outputRow[outColIndex] = biasView[outChIndex]
					}

					for inChIndex := range inChannels {
						inputChannelOffset := inputBatchOffset + inChIndex*inCStride
						weightChannelOffset := weightOutOffset + inChIndex*weightCStride

						for kDIndex := range kD {
							inputPlaneOffset := inputChannelOffset + (outDIndex+kDIndex)*inDStride
							weightPlaneOffset := weightChannelOffset + kDIndex*weightDStride

							if blockCols > 0 {
								conv2dStride1RowNEONAsm(
									&rowScratch[0],
									&inputView[inputPlaneOffset],
									&weightView[weightPlaneOffset],
									0,
									blockCols,
									1, kH, kW,
									inHStride, inDStride,
									weightHStride, weightDStride,
									outHIndex, 0,
								)
							}

							for outColIndex := blockCols; outColIndex < outW; outColIndex++ {
								rowScratch[outColIndex] = conv3DPlanePixelScalar(
									inputView, weightView,
									inputPlaneOffset, weightPlaneOffset,
									inH, inW, kH, kW,
									outHIndex, outColIndex,
								)
							}

							addFloat32Native(outputRow, outputRow, rowScratch)
						}
					}
				}
			}
		}
	}
}

func conv3DConfigNEONEligible(config Conv3DConfig) bool {
	return config.StrideD == 1 &&
		config.StrideH == 1 &&
		config.StrideW == 1 &&
		config.PaddingD == 0 &&
		config.PaddingH == 0 &&
		config.PaddingW == 0 &&
		config.DilationD == 1 &&
		config.DilationH == 1 &&
		config.DilationW == 1
}

func conv3DPlanePixelScalar(
	inputView, weightView []float32,
	inputPlaneOffset, weightPlaneOffset int,
	inH, inW, kH, kW, outHIndex, outColIndex int,
) float32 {
	sum := float32(0)

	for kHIndex := range kH {
		for kWIndex := range kW {
			inRow := outHIndex + kHIndex
			inCol := outColIndex + kWIndex

			if inRow < 0 || inRow >= inH || inCol < 0 || inCol >= inW {
				continue
			}

			inputIndex := inputPlaneOffset + inRow*inW + inCol
			weightIndex := weightPlaneOffset + kHIndex*kW + kWIndex
			sum += inputView[inputIndex] * weightView[weightIndex]
		}
	}

	return sum
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
