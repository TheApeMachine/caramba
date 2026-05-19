//go:build !arm64

package cpu

func AdaptivePool2DFloat32Native(
	inputView, outputView []float32,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	adaptivePool2DFloat32Scalar(
		inputView, outputView,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
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
