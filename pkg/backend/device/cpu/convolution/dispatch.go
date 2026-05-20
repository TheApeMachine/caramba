package convolution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func dispatchConv2D(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
	f32Native func(
		Conv2DConfig,
		unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer,
		int, int, int, int, int, int, int, int, int,
	),
) {
	inputLength := batch * inChannels * inHeight * inWidth
	weightLength := outChannels * inChannels * kernelHeight * kernelWidth
	biasLength := outChannels
	outputLength := batch * outChannels * outHeight * outWidth

	dispatchConv4(
		input, weight, bias, output,
		inputLength, weightLength, biasLength, outputLength,
		format,
		func(
			inputPointer, weightPointer, biasPointer, outputPointer unsafe.Pointer,
		) {
			f32Native(
				config,
				inputPointer, weightPointer, biasPointer, outputPointer,
				batch, inChannels, inHeight, inWidth,
				outChannels, kernelHeight, kernelWidth,
				outHeight, outWidth,
			)
		},
	)
}

func dispatchConv1D(
	config Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
	format dtype.DType,
	f32Native func(
		Conv1DConfig,
		unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer,
		int, int, int, int, int, int,
	),
) {
	inputLength := batch * inChannels * inLength
	weightLength := outChannels * inChannels * kernelLength
	biasLength := outChannels
	outputLength := batch * outChannels * outLength

	dispatchConv4(
		input, weight, bias, output,
		inputLength, weightLength, biasLength, outputLength,
		format,
		func(
			inputPointer, weightPointer, biasPointer, outputPointer unsafe.Pointer,
		) {
			f32Native(
				config,
				inputPointer, weightPointer, biasPointer, outputPointer,
				batch, inChannels, inLength, outChannels, kernelLength, outLength,
			)
		},
	)
}

func dispatchConv3D(
	config Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
	format dtype.DType,
	f32Native func(
		Conv3DConfig,
		unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer,
		int, int, int, int, int, int, int, int, int, int, int, int,
	),
) {
	inputLength := batch * inChannels * inD * inH * inW
	weightLength := outChannels * inChannels * kD * kH * kW
	biasLength := outChannels
	outputLength := batch * outChannels * outD * outH * outW

	dispatchConv4(
		input, weight, bias, output,
		inputLength, weightLength, biasLength, outputLength,
		format,
		func(
			inputPointer, weightPointer, biasPointer, outputPointer unsafe.Pointer,
		) {
			f32Native(
				config,
				inputPointer, weightPointer, biasPointer, outputPointer,
				batch, inChannels, inD, inH, inW,
				outChannels, kD, kH, kW, outD, outH, outW,
			)
		},
	)
}

func dispatchConvTranspose2D(
	config Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
	f32Native func(
		Conv2DConfig,
		unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer,
		int, int, int, int, int, int, int, int, int,
	),
) {
	dispatchConv2D(
		config,
		input, weight, bias, output,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
		f32Native,
	)
}

func dispatchConv4(
	input, weight, bias, output unsafe.Pointer,
	inputLength, weightLength, biasLength, outputLength int,
	format dtype.DType,
	f32Native func(
		inputPointer, weightPointer, biasPointer, outputPointer unsafe.Pointer,
	),
) {
	if inputLength == 0 || outputLength == 0 {
		return
	}

	if format == dtype.Float32 {
		f32Native(input, weight, bias, output)
		return
	}

	inputF32 := BorrowFloat32Buffer(inputLength)
	weightF32 := BorrowFloat32Buffer(weightLength)
	biasF32 := BorrowFloat32Buffer(biasLength)
	outputF32 := BorrowFloat32Buffer(outputLength)

	defer ReleaseFloat32Buffer(inputF32)
	defer ReleaseFloat32Buffer(weightF32)
	defer ReleaseFloat32Buffer(biasF32)
	defer ReleaseFloat32Buffer(outputF32)

	widenToF32Buffer(
		unsafe.Pointer(unsafe.SliceData(inputF32)),
		input,
		inputLength,
		format,
	)
	widenToF32Buffer(
		unsafe.Pointer(unsafe.SliceData(weightF32)),
		weight,
		weightLength,
		format,
	)
	widenToF32Buffer(
		unsafe.Pointer(unsafe.SliceData(biasF32)),
		bias,
		biasLength,
		format,
	)

	f32Native(
		unsafe.Pointer(unsafe.SliceData(inputF32)),
		unsafe.Pointer(unsafe.SliceData(weightF32)),
		unsafe.Pointer(unsafe.SliceData(biasF32)),
		unsafe.Pointer(unsafe.SliceData(outputF32)),
	)

	narrowFromF32Buffer(
		output,
		unsafe.Pointer(unsafe.SliceData(outputF32)),
		outputLength,
		format,
	)
}
