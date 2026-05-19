//go:build arm64

package pool

import (
	"unsafe"

	"github.com/theapemachine/caramba/pkg/backend/device/cpu/reduction"
)

func Pool2DFloat32Native(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	inputLength := batch * channels * inHeight * inWidth
	outputLength := batch * channels * outHeight * outWidth
	inputView := float32View(input, inputLength)
	outputView := float32View(output, outputLength)

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

				if blockCols > 0 && strideTwo && useMax {
					MaxPool2x2Stride2RowNEONAsm(
						&outputRow[0], &channel[0],
						blockCols, inWidth, ihStart,
					)
				}

				if blockCols > 0 && strideTwo && !useMax {
					AvgPool2x2Stride2RowNEONAsm(
						&outputRow[0], &channel[0],
						blockCols, inWidth, ihStart,
					)
				}

				if blockCols > 0 && !strideTwo {
					if useMax {
						MaxPool2DStride1RowNEONAsm(
							&outputRow[0], &channel[0],
							blockCols, config.KernelH, config.KernelW,
							inWidth, ihStart,
						)
					}

					if !useMax {
						AvgPool2DStride1RowNEONAsm(
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

func PoolWindowMaxFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	elementCount := (endRow - startRow) * (endCol - startCol)

	if elementCount <= 0 {
		return PoolWindowMaxScalar(channel, inWidth, startRow, endRow, startCol, endCol)
	}

	scratch := BorrowFloat32Buffer(elementCount)
	defer ReleaseFloat32Buffer(scratch)

	PoolWindowGather(channel, scratch, inWidth, startRow, endRow, startCol, endCol)

	return reduction.ReduceMaxFloat32Native(scratch[:elementCount])
}

func PoolWindowAvgFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	elementCount := (endRow - startRow) * (endCol - startCol)

	if elementCount <= 0 {
		return PoolWindowAvgScalar(channel, inWidth, startRow, endRow, startCol, endCol)
	}

	scratch := BorrowFloat32Buffer(elementCount)
	defer ReleaseFloat32Buffer(scratch)

	PoolWindowGather(channel, scratch, inWidth, startRow, endRow, startCol, endCol)

	return sumFloat32Sequential(scratch[:elementCount]) / float32(elementCount)
}

func AdaptivePool2DFloat32Native(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	inputLength := batch * channels * inHeight * inWidth
	outputLength := batch * channels * outHeight * outWidth
	inputView := float32View(input, inputLength)
	outputView := float32View(output, outputLength)

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
