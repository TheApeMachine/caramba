package pool

import (
	"math"
	"unsafe"
)

func Pool2DFloat32Scalar(
	config PoolConfig,
	inputView, outputView []float32,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for channelIndex := 0; channelIndex < channels; channelIndex++ {
			channelOffsetIn := (batchIndex*channels + channelIndex) * inHeight * inWidth
			channelOffsetOut := (batchIndex*channels + channelIndex) * outHeight * outWidth

			for outRow := 0; outRow < outHeight; outRow++ {
				for outCol := 0; outCol < outWidth; outCol++ {
					value := poolWindow(
						inputView[channelOffsetIn:channelOffsetIn+inHeight*inWidth],
						inHeight, inWidth,
						outRow, outCol,
						config, useMax,
					)

					outputView[channelOffsetOut+outRow*outWidth+outCol] = value
				}
			}
		}
	}
}

func poolWindow(
	channel []float32,
	inHeight, inWidth int,
	outRow, outCol int,
	config PoolConfig,
	useMax bool,
) float32 {
	startRow := outRow*config.StrideH - config.PaddingH
	startCol := outCol*config.StrideW - config.PaddingW

	value := float32(math.Inf(-1))

	if !useMax {
		value = 0
	}

	count := 0

	for kernelRow := 0; kernelRow < config.KernelH; kernelRow++ {
		for kernelCol := 0; kernelCol < config.KernelW; kernelCol++ {
			row := startRow + kernelRow
			col := startCol + kernelCol

			if row < 0 || row >= inHeight || col < 0 || col >= inWidth {
				continue
			}

			candidate := channel[row*inWidth+col]
			count++

			switch {
			case useMax:
				if candidate > value {
					value = candidate
				}
			default:
				value += candidate
			}
		}
	}

	if !useMax && count > 0 {
		value /= float32(count)
	}

	return value
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

func float32View(pointer unsafe.Pointer, length int) []float32 {
	return unsafe.Slice((*float32)(pointer), length)
}
