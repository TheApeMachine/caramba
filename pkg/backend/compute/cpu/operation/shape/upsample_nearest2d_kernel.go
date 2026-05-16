package shape

func upsampleNearest2DGenericKernel(
	dst []float64,
	src []float64,
	batch int,
	channels int,
	height int,
	width int,
	scaleH int,
	scaleW int,
) {
	outputHeight := height * scaleH
	outputWidth := width * scaleW
	inputImage := channels * height * width
	outputImage := channels * outputHeight * outputWidth

	for batchIndex := range batch {
		inputBatch := batchIndex * inputImage
		outputBatch := batchIndex * outputImage

		for channelIndex := range channels {
			inputChannel := inputBatch + channelIndex*height*width
			outputChannel := outputBatch + channelIndex*outputHeight*outputWidth

			for rowIndex := range height {
				sourceRow := src[inputChannel+rowIndex*width : inputChannel+(rowIndex+1)*width]
				targetRowOffset := outputChannel + rowIndex*scaleH*outputWidth
				targetRow := dst[targetRowOffset : targetRowOffset+outputWidth]

				upsampleNearest2DRowKernel(targetRow, sourceRow, scaleW)

				for repeatIndex := 1; repeatIndex < scaleH; repeatIndex++ {
					repeatedRowOffset := targetRowOffset + repeatIndex*outputWidth
					reshapeKernel(
						dst[repeatedRowOffset:repeatedRowOffset+outputWidth],
						targetRow,
					)
				}
			}
		}
	}
}

func upsampleNearest2DRowGenericKernel(dst []float64, src []float64, scaleW int) {
	for sourceIndex, value := range src {
		targetOffset := sourceIndex * scaleW

		for repeatIndex := range scaleW {
			dst[targetOffset+repeatIndex] = value
		}
	}
}
