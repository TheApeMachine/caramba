//go:build 386

package pool

import "unsafe"

func Pool2DFloat32Native(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	inputLength := batch * channels * inHeight * inWidth
	outputLength := batch * channels * outHeight * outWidth

	Pool2DFloat32Scalar(
		config,
		float32View(input, inputLength),
		float32View(output, outputLength),
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}

func PoolWindowMaxFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	return PoolWindowMaxScalar(channel, inWidth, startRow, endRow, startCol, endCol)
}

func PoolWindowAvgFloat32Native(
	channel []float32,
	inWidth, startRow, endRow, startCol, endCol int,
) float32 {
	return PoolWindowAvgScalar(channel, inWidth, startRow, endRow, startCol, endCol)
}

func AdaptivePool2DFloat32Native(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	inputLength := batch * channels * inHeight * inWidth
	outputLength := batch * channels * outHeight * outWidth

	AdaptivePool2DFloat32Scalar(
		float32View(input, inputLength),
		float32View(output, outputLength),
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}
