package pool

func AdaptivePool2DFloat32Scalar(
	inputView, outputView []float32,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	for batchIndex := range batch {
		for channelIndex := range channels {
			for outRow := range outHeight {
				startRow := (outRow * inHeight) / outHeight
				endRow := ((outRow + 1) * inHeight) / outHeight

				for outCol := range outWidth {
					startCol := (outCol * inWidth) / outWidth
					endCol := ((outCol + 1) * inWidth) / outWidth

					value := outputAdaptivePoolValue(
						inputView, batchIndex, channelIndex, channels,
						inHeight, inWidth, startRow, endRow, startCol, endCol, useMax,
					)

					outputView[((batchIndex*channels+channelIndex)*outHeight+outRow)*outWidth+outCol] = value
				}
			}
		}
	}
}

func outputAdaptivePoolValue(
	inputView []float32,
	batchIndex, channelIndex, channels, inHeight, inWidth int,
	startRow, endRow, startCol, endCol int,
	useMax bool,
) float32 {
	var sum float32
	maximum := float32(-1e30)
	count := 0

	for row := startRow; row < endRow; row++ {
		for col := startCol; col < endCol; col++ {
			value := inputView[((batchIndex*channels+channelIndex)*inHeight+row)*inWidth+col]
			count++

			if useMax {
				if value > maximum {
					maximum = value
				}

				continue
			}

			sum += value
		}
	}

	if useMax {
		return maximum
	}

	if count == 0 {
		return 0
	}

	return sum / float32(count)
}
