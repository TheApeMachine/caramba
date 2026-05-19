package convolution

import "unsafe"

type conv2DF32Runner func(
	Conv2DConfig,
	unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer,
	int, int, int, int, int, int, int, int, int,
)

type conv1DF32Runner func(
	Conv1DConfig,
	unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer,
	int, int, int, int, int, int,
)

type conv3DF32Runner func(
	Conv3DConfig,
	unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer,
	int, int, int, int, int, int, int, int, int, int, int, int,
)

var (
	conv2DF32Kernel          conv2DF32Runner = Conv2DFloat32Native
	conv1DF32Kernel          conv1DF32Runner = Conv1DFloat32Native
	conv3DF32Kernel          conv3DF32Runner = Conv3DFloat32Native
	convTranspose2DF32Kernel conv2DF32Runner = ConvTranspose2DFloat32Native
)
