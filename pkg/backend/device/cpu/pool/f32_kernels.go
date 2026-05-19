package pool

import "unsafe"

type pool2DF32Runner func(
	PoolConfig,
	unsafe.Pointer, unsafe.Pointer,
	int, int, int, int, int, int,
	bool,
)

type adaptivePool2DF32Runner func(
	unsafe.Pointer, unsafe.Pointer,
	int, int, int, int, int, int,
	bool,
)

var (
	pool2DF32Kernel         pool2DF32Runner         = Pool2DFloat32Native
	adaptivePool2DF32Kernel adaptivePool2DF32Runner = AdaptivePool2DFloat32Native
)
