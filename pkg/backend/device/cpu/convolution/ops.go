package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func Conv2D(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
) {
	dispatchConv2D(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
		runConv2DF32,
	)
}

func Conv1D(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
	format dtype.DType,
) {
	dispatchConv1D(
		config,
		input, weight, bias, output,
		batch, inChannels, inLength, outChannels, kernelLength, outLength,
		format,
		runConv1DF32,
	)
}

func Conv3D(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
	format dtype.DType,
) {
	dispatchConv3D(
		config,
		input, weight, bias, output,
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW,
		format,
		runConv3DF32,
	)
}

func ConvTranspose2D(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
) {
	dispatchConvTranspose2D(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
		runConvTranspose2DF32,
	)
}
