//go:build arm64

package kernels

func pool2DFloat32Native(
	config PoolConfig,
	inputView, outputView []float32,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	if poolConfigNEONEligible(config) {
		pool2DFloat32FastRowNative(
			config, inputView, outputView,
			batch, channels, inHeight, inWidth, outHeight, outWidth,
			useMax,
		)

		return
	}

	pool2DFloat32WindowNative(
		config, inputView, outputView,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}

func pool2DFloat32FastRowNative(
	config PoolConfig,
	inputView, outputView []float32,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	strideTwo := config.StrideH == 2 && config.StrideW == 2

	for batchIndex := range batch {
		for channelIndex := range channels {
			channelOffsetIn := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channelOffsetOut := (batchIndex*channels + channelIndex) * outHeight * outWidth
			channel := inputView[channelOffsetIn : channelOffsetIn+inHeight*inWidth]
			outChannel := outputView[channelOffsetOut : channelOffsetOut+outHeight*outWidth]

			for outRow := range outHeight {
				outputRow := outChannel[outRow*outWidth : (outRow+1)*outWidth]
				blockCols := len(outputRow) &^ 3
				ihStart := outRow*config.StrideH - config.PaddingH

				if blockCols > 0 && strideTwo {
					if useMax {
						maxPool2x2Stride2RowNEONAsm(
							&outputRow[0], &channel[0],
							blockCols, inWidth, ihStart,
						)
					}

					if !useMax {
						avgPool2x2Stride2RowNEONAsm(
							&outputRow[0], &channel[0],
							blockCols, inWidth, ihStart,
						)
					}
				}

				if blockCols > 0 && !strideTwo {
					if useMax {
						maxPool2DStride1RowNEONAsm(
							&outputRow[0], &channel[0],
							blockCols, config.KernelH, config.KernelW,
							inWidth, ihStart,
						)
					}

					if !useMax {
						avgPool2DStride1RowNEONAsm(
							&outputRow[0], &channel[0],
							blockCols, config.KernelH, config.KernelW,
							inWidth, ihStart,
						)
					}
				}

				for outCol := blockCols; outCol < outWidth; outCol++ {
					outputRow[outCol] = poolWindow(
						channel, inHeight, inWidth,
						outRow, outCol, config, useMax,
					)
				}
			}
		}
	}
}

func pool2DFloat32WindowNative(
	config PoolConfig,
	inputView, outputView []float32,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	for batchIndex := range batch {
		for channelIndex := range channels {
			channelOffsetIn := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channelOffsetOut := (batchIndex*channels + channelIndex) * outHeight * outWidth
			channel := inputView[channelOffsetIn : channelOffsetIn+inHeight*inWidth]

			for outRow := range outHeight {
				for outCol := range outWidth {
					outputIndex := channelOffsetOut + outRow*outWidth + outCol

					if !poolWindowFullyInBounds(
						inHeight, inWidth, outRow, outCol, config,
					) {
						outputView[outputIndex] = poolWindow(
							channel, inHeight, inWidth,
							outRow, outCol, config, useMax,
						)

						continue
					}

					startRow := outRow*config.StrideH - config.PaddingH
					startCol := outCol*config.StrideW - config.PaddingW
					endRow := startRow + config.KernelH
					endCol := startCol + config.KernelW

					if useMax {
						outputView[outputIndex] = poolWindowMaxFloat32Native(
							channel, inWidth,
							startRow, endRow, startCol, endCol,
						)

						continue
					}

					outputView[outputIndex] = poolWindowAvgFloat32Native(
						channel, inWidth,
						startRow, endRow, startCol, endCol,
					)
				}
			}
		}
	}
}

func poolWindowFullyInBounds(
	inHeight, inWidth, outRow, outCol int,
	config PoolConfig,
) bool {
	startRow := outRow*config.StrideH - config.PaddingH
	startCol := outCol*config.StrideW - config.PaddingW

	if startRow < 0 || startCol < 0 {
		return false
	}

	endRow := startRow + config.KernelH
	endCol := startCol + config.KernelW

	return endRow <= inHeight && endCol <= inWidth
}

func poolConfigNEONEligible(config PoolConfig) bool {
	if config.PaddingH != 0 || config.PaddingW != 0 {
		return false
	}

	if config.StrideH == 1 && config.StrideW == 1 {
		return true
	}

	return config.KernelH == 2 &&
		config.KernelW == 2 &&
		config.StrideH == 2 &&
		config.StrideW == 2
}
