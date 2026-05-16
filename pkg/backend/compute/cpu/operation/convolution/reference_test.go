package convolution

import . "github.com/smartystreets/goconvey/convey"

func convolutionSequence(length int, scale, offset float64) []float64 {
	values := make([]float64, length)

	for index := range values {
		pattern := float64((index % 17) - 8)
		values[index] = offset + scale*pattern
	}

	return values
}

func assertConvolutionAlmostEqual(actual, expected []float64) {
	So(actual, ShouldHaveLength, len(expected))

	for index := range expected {
		So(actual[index], ShouldAlmostEqual, expected[index], 1e-9)
	}
}

func referenceConv1d(
	input, weight, bias []float64,
	batch, inChannels, length int,
	outChannels, kernelSize, stride, padding, dilation, groups int,
) []float64 {
	lengthOut := (length+2*padding-dilation*(kernelSize-1)-1)/stride + 1
	output := make([]float64, batch*outChannels*lengthOut)
	inChannelsPerGroup := inChannels / groups
	outChannelsPerGroup := outChannels / groups

	for batchIndex := range batch {
		for groupIndex := range groups {
			inChannelStart := groupIndex * inChannelsPerGroup
			outChannelStart := groupIndex * outChannelsPerGroup

			for outChannelOffset := range outChannelsPerGroup {
				outChannel := outChannelStart + outChannelOffset

				for outputIndex := range lengthOut {
					sum := bias[outChannel]

					for inChannelOffset := range inChannelsPerGroup {
						inputChannel := inChannelStart + inChannelOffset

						for kernelIndex := range kernelSize {
							inputIndex := outputIndex*stride + kernelIndex*dilation - padding

							if inputIndex < 0 || inputIndex >= length {
								continue
							}

							inputOffset := (batchIndex*inChannels+inputChannel)*length + inputIndex
							weightOffset := (outChannel*inChannelsPerGroup+inChannelOffset)*kernelSize + kernelIndex
							sum += input[inputOffset] * weight[weightOffset]
						}
					}

					output[(batchIndex*outChannels+outChannel)*lengthOut+outputIndex] = sum
				}
			}
		}
	}

	return output
}

func referenceConv2d(
	input, weight, bias []float64,
	batch, inChannels, height, width int,
	outChannels, kernelH, kernelW int,
	strideH, strideW, padH, padW int,
	dilationH, dilationW, groups int,
) []float64 {
	heightOut := (height+2*padH-dilationH*(kernelH-1)-1)/strideH + 1
	widthOut := (width+2*padW-dilationW*(kernelW-1)-1)/strideW + 1
	output := make([]float64, batch*outChannels*heightOut*widthOut)
	inChannelsPerGroup := inChannels / groups
	outChannelsPerGroup := outChannels / groups

	for batchIndex := range batch {
		for groupIndex := range groups {
			inChannelStart := groupIndex * inChannelsPerGroup
			outChannelStart := groupIndex * outChannelsPerGroup

			for outChannelOffset := range outChannelsPerGroup {
				outChannel := outChannelStart + outChannelOffset

				for outputRow := range heightOut {
					for outputColumn := range widthOut {
						sum := bias[outChannel]

						for inChannelOffset := range inChannelsPerGroup {
							inputChannel := inChannelStart + inChannelOffset

							for kernelRow := range kernelH {
								inputRow := outputRow*strideH + kernelRow*dilationH - padH

								if inputRow < 0 || inputRow >= height {
									continue
								}

								for kernelColumn := range kernelW {
									inputColumn := outputColumn*strideW + kernelColumn*dilationW - padW

									if inputColumn < 0 || inputColumn >= width {
										continue
									}

									inputOffset := ((batchIndex*inChannels+inputChannel)*height+inputRow)*width + inputColumn
									weightOffset := ((outChannel*inChannelsPerGroup+inChannelOffset)*kernelH+kernelRow)*kernelW + kernelColumn
									sum += input[inputOffset] * weight[weightOffset]
								}
							}
						}

						outputOffset := ((batchIndex*outChannels+outChannel)*heightOut+outputRow)*widthOut + outputColumn
						output[outputOffset] = sum
					}
				}
			}
		}
	}

	return output
}

func referenceConv3d(
	input, weight, bias []float64,
	batch, inChannels, depth, height, width int,
	outChannels, kernelD, kernelH, kernelW int,
	strideD, strideH, strideW int,
	padD, padH, padW int,
	dilationD, dilationH, dilationW, groups int,
) []float64 {
	depthOut := (depth+2*padD-dilationD*(kernelD-1)-1)/strideD + 1
	heightOut := (height+2*padH-dilationH*(kernelH-1)-1)/strideH + 1
	widthOut := (width+2*padW-dilationW*(kernelW-1)-1)/strideW + 1
	output := make([]float64, batch*outChannels*depthOut*heightOut*widthOut)
	inChannelsPerGroup := inChannels / groups
	outChannelsPerGroup := outChannels / groups

	for batchIndex := range batch {
		for groupIndex := range groups {
			inChannelStart := groupIndex * inChannelsPerGroup
			outChannelStart := groupIndex * outChannelsPerGroup

			for outChannelOffset := range outChannelsPerGroup {
				outChannel := outChannelStart + outChannelOffset

				for outputDepth := range depthOut {
					for outputRow := range heightOut {
						for outputColumn := range widthOut {
							sum := bias[outChannel]

							for inChannelOffset := range inChannelsPerGroup {
								inputChannel := inChannelStart + inChannelOffset

								for kernelDepth := range kernelD {
									inputDepth := outputDepth*strideD + kernelDepth*dilationD - padD

									if inputDepth < 0 || inputDepth >= depth {
										continue
									}

									for kernelRow := range kernelH {
										inputRow := outputRow*strideH + kernelRow*dilationH - padH

										if inputRow < 0 || inputRow >= height {
											continue
										}

										for kernelColumn := range kernelW {
											inputColumn := outputColumn*strideW + kernelColumn*dilationW - padW

											if inputColumn < 0 || inputColumn >= width {
												continue
											}

											inputOffset := (((batchIndex*inChannels+inputChannel)*depth+inputDepth)*height+inputRow)*width + inputColumn
											weightOffset := (((outChannel*inChannelsPerGroup+inChannelOffset)*kernelD+kernelDepth)*kernelH+kernelRow)*kernelW + kernelColumn
											sum += input[inputOffset] * weight[weightOffset]
										}
									}
								}
							}

							outputOffset := (((batchIndex*outChannels+outChannel)*depthOut+outputDepth)*heightOut+outputRow)*widthOut + outputColumn
							output[outputOffset] = sum
						}
					}
				}
			}
		}
	}

	return output
}

func referenceConvTranspose2d(
	input, weight, bias []float64,
	batch, inChannels, height, width int,
	outChannels, kernelH, kernelW int,
	strideH, strideW, groups int,
) []float64 {
	heightOut := (height-1)*strideH + kernelH
	widthOut := (width-1)*strideW + kernelW
	output := make([]float64, batch*outChannels*heightOut*widthOut)
	inChannelsPerGroup := inChannels / groups
	outChannelsPerGroup := outChannels / groups

	for batchIndex := range batch {
		for outChannel := range outChannels {
			base := (batchIndex*outChannels + outChannel) * heightOut * widthOut

			for index := range heightOut * widthOut {
				output[base+index] = bias[outChannel]
			}
		}
	}

	for batchIndex := range batch {
		for groupIndex := range groups {
			inputChannelStart := groupIndex * inChannelsPerGroup
			outputChannelStart := groupIndex * outChannelsPerGroup

			for inputChannelOffset := range inChannelsPerGroup {
				inputChannel := inputChannelStart + inputChannelOffset
				weightChannelBase := inputChannel * outChannelsPerGroup * kernelH * kernelW

				for inputRow := range height {
					for inputColumn := range width {
						inputOffset := ((batchIndex*inChannels+inputChannel)*height+inputRow)*width + inputColumn
						inputValue := input[inputOffset]

						for outputChannelOffset := range outChannelsPerGroup {
							outputChannel := outputChannelStart + outputChannelOffset

							for kernelRow := range kernelH {
								outputRow := inputRow*strideH + kernelRow

								for kernelColumn := range kernelW {
									outputColumn := inputColumn*strideW + kernelColumn
									weightOffset := weightChannelBase + (outputChannelOffset*kernelH+kernelRow)*kernelW + kernelColumn
									outputOffset := ((batchIndex*outChannels+outputChannel)*heightOut+outputRow)*widthOut + outputColumn
									output[outputOffset] += inputValue * weight[weightOffset]
								}
							}
						}
					}
				}
			}
		}
	}

	return output
}
