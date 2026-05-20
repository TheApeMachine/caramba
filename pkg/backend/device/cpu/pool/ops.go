package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func MaxPool2D(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	dispatchPool2D(
		config, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format, true, runMaxPool2DF32,
	)
}

func AvgPool2D(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	dispatchPool2D(
		config, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format, false, runAvgPool2DF32,
	)
}

func AdaptiveMaxPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	dispatchAdaptivePool2D(
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format, true, runAdaptiveMaxPool2DF32,
	)
}

func AdaptiveAvgPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	dispatchAdaptivePool2D(
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		format, false, runAdaptiveAvgPool2DF32,
	)
}

func dispatchPool2D(
	config PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
	useMax bool,
	f32 func(
		PoolConfig,
		unsafe.Pointer, unsafe.Pointer,
		int, int, int, int, int, int,
		bool,
	),
) {
	if format != dtype.Float32 {
		panic("pool: only dtype.Float32 is implemented")
	}

	f32(
		config, input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}

func dispatchAdaptivePool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
	useMax bool,
	f32 func(
		unsafe.Pointer, unsafe.Pointer,
		int, int, int, int, int, int,
		bool,
	),
) {
	if format != dtype.Float32 {
		panic("pool: only dtype.Float32 is implemented")
	}

	f32(
		input, output,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)
}
