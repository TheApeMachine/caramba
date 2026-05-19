//go:build arm64

package cpu

func AdaptivePool2DFloat32Native(
	inputView, outputView []float32,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	for batchIndex := range batch {
		for channelIndex := range channels {
			channelOffset := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channel := inputView[channelOffset : channelOffset+inHeight*inWidth]
			outputOffset := (batchIndex*channels + channelIndex) * outHeight * outWidth

			for outRow := range outHeight {
				startRow := (outRow * inHeight) / outHeight
				endRow := ((outRow + 1) * inHeight) / outHeight

				for outCol := range outWidth {
					startCol := (outCol * inWidth) / outWidth
					endCol := ((outCol + 1) * inWidth) / outWidth
					outputIndex := outputOffset + outRow*outWidth + outCol

					if useMax {
						outputView[outputIndex] = PoolWindowMaxFloat32Native(
							channel, inWidth,
							startRow, endRow, startCol, endCol,
						)

						continue
					}

					outputView[outputIndex] = PoolWindowAvgFloat32Native(
						channel, inWidth,
						startRow, endRow, startCol, endCol,
					)
				}
			}
		}
	}
}

func adaptivePool2DFloat32Scalar(
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

					value := neon.OutputAdaptivePoolValue(
						inputView, batchIndex, channelIndex, channels,
						inHeight, inWidth, startRow, endRow, startCol, endCol, useMax,
					)

					outputView[((batchIndex*channels+channelIndex)*outHeight+outRow)*outWidth+outCol] = value
				}
			}
		}
	}
}
