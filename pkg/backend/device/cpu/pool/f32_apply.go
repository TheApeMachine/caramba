package pool

import "unsafe"

func runMaxPool2DF32(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	pool2DF32Kernel(
		config, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}

func runAvgPool2DF32(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	pool2DF32Kernel(
		config, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}

func runAdaptiveMaxPool2DF32(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	adaptivePool2DF32Kernel(
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}

func runAdaptiveAvgPool2DF32(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	useMax bool,
) {
	adaptivePool2DF32Kernel(
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}
